require 'torch'
require 'nn'
require 'nngraph'
require 'loadcaffe'
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'
require 'misc.DataLoader'
require 'misc.LanguageModel'
-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Person Search with Natural Language Description')
cmd:text()
cmd:text('Options')

-- Input paths
cmd:option('-model','snapshot/lstm1_rnn512_bestACC.t7','path to model to evaluate')
-- Basic options
cmd:option('-batch_size', 1, 'if > 0 then overrule, otherwise load from checkpoint.')
cmd:option('-num_images', -1, 'how many images to use when periodically evaluating the loss? (-1 = all)')

cmd:option('-input_h5','../data/reidtalk.h5','path to the h5file containing the preprocessed dataset. empty = fetch from model checkpoint.')
cmd:option('-input_json','../data/reidtalk.json','path to the json file containing additional info and vocab. empty = fetch from model checkpoint.')
cmd:option('-split', 'test', 'if running on MSCOCO images, which split to use: val|test|train')
-- misc
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-id', 'evalscript', 'an id identifying this run/job. used only if language_eval = 1 for appending to intermediate files')
cmd:option('-seed', 123, 'random number generator seed to use')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
cmd:text()

-------------------------------------------------------------------------------
-- Basic Torch initializations
-------------------------------------------------------------------------------
local opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

if opt.gpuid >= 0 then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then require 'cudnn' end
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpuid + 1) -- note +1 because lua is 1-indexed
end
-------------------------------------------------------------------------------
-- Load the model checkpoint to evaluate
-------------------------------------------------------------------------------
assert(string.len(opt.model) > 0, 'must provide a model')
local checkpoint = torch.load(opt.model)
-- override and collect parameters
if string.len(opt.input_h5) == 0 then opt.input_h5 = checkpoint.opt.input_h5 end
if string.len(opt.input_json) == 0 then opt.input_json = checkpoint.opt.input_json end
if opt.batch_size == 0 then opt.batch_size = checkpoint.opt.batch_size end
local fetch = {'rnn_size', 'input_encoding_size', 'drop_prob_lm', 'cnn_proto', 'cnn_model'}
for k,v in pairs(fetch) do 
  opt[v] = checkpoint.opt[v] -- copy over options from model
end
local vocab = checkpoint.vocab -- ix -> word mapping
print(opt)
-------------------------------------------------------------------------------
-- Create the Data Loader instance
-------------------------------------------------------------------------------
local loader = DataLoader{h5_file = opt.input_h5, json_file = opt.input_json}
-------------------------------------------------------------------------------
-- Load the networks from model checkpoint
-------------------------------------------------------------------------------
local protos = checkpoint.protos
protos.crit = nn.BCECriterion()
protos.lm:createClones() -- reconstruct clones inside the language model
if opt.gpuid >= 0 then for k,v in pairs(protos) do v:cuda() end end

-------------------------------------------------------------------------------
-- Extract image features
-------------------------------------------------------------------------------
local function ExtractImg(split, evalopt)
  local verbose = utils.getopt(evalopt, 'verbose', true)
  local num_images = utils.getopt(evalopt, 'num_images', true)
  protos.cnn:evaluate()
  protos.lm:evaluate()
  loader:resetIterator(split)
  local n = 0
  TestData = {}

  while true do
    local data = loader:getBatch{batch_size = opt.batch_size, split = split, seq_per_img = 2}

    data.images = net_utils.prepro(data.images, false, opt.gpuid >= 0) 
    local feats = protos.cnn:forward(data.images)
    data.feat = torch.Tensor(feats:size())
    data.feat:copy(feats)
    table.insert(TestData, data)

    n = n + 1

    -- if we wrapped around the split or used up val imgs budget then bail
    local ix0 = data.bounds.it_pos_now
    local ix1 = math.min(data.bounds.it_max, num_images)
    if n % 100 == 0 then print(string.format('evaluating performance... %d/%d', ix0-1, ix1)) end

    --if n==100 then break end
    if data.bounds.wrapped then break end -- the split ran out of data, lets break out
    if num_images >= 0 and n >= num_images then break end -- we've used enough images
  end

  print(#TestData)
end

-------------------------------------------------------------
-- Retrieval
-------------------------------------------------------------
local function Retrieval(split, evalopt)

  local kvals = {1, 5, 10} --{1, 5, 10, 50, #TestData}
  local correct = {}
  local n = 0

  -------------------------------------------------------------
  -- seperate the data into multiple splits to avoid memory leak
  -------------------------------------------------------------
  local split1 = 500
  local nsplit = math.floor(#TestData/split1)+1
  local split2 = #TestData - (nsplit-1)*split1
  local img_dim = TestData[1].feat:size(2)
  local Gfeats = {}
  local ids = torch.zeros(#TestData)
  local count = 1

  for i=1,nsplit do
    local split_size = split1
    if i==nsplit then split_size = split2 end
    if split_size==0 then break end

    local Gfeat = torch.CudaTensor(split_size, img_dim) 
    for j=1,split_size do
      Gfeat[j] = TestData[count].feat
      ids[count] = TestData[count].infos[1].id
      count = count + 1
    end

    table.insert(Gfeats, Gfeat)
  end

  -------------------------------------------------------------
  -- txt2img
  -------------------------------------------------------------
  for k, Query in pairs (TestData) do    
    
    for iSent = 1,Query.labels:size(2) do
    
      local Qlabel = Query.labels:narrow(2,iSent,1)
      local Qseqlen = Query.seqlen[iSent]
      local Qid = Query.infos[1].id
      local losses = torch.zeros(#TestData, 1)
      local count2 = 1
      -------------------------------------------------------------
      -- seperate the data into 2 splits to avoid memory leak
      -------------------------------------------------------------
      for i=1,nsplit do
        local split_size = split1
        if i==nsplit then split_size = split2 end
        if split_size==0 then break end

        local Qlabel_i = torch.expand(Qlabel, Qlabel:size(1), split_size)
        local Qseqlen_i = torch.Tensor(split_size):fill(Qseqlen)
        local logprobs_i = protos.lm:forward{Gfeats[i], Qlabel_i, Qseqlen_i}
        losses:narrow(1,count2,split_size):copy(-logprobs_i)
        count2 = count2 + split_size
      end

      local _, indexes = torch.sort(losses,1)
      
      for _,kval in pairs(kvals) do
        if not correct[kval] then correct[kval] = 0 end
        for i=1,kval do
          if Qid==ids[indexes[i][1]] then correct[kval]=correct[kval]+1 break end
        end
      end

      n = n + 1
    end
    if k%10==0 then print(string.format('testing... %d/%d', k, #TestData)) end
  end

  assert(n==#TestData*2, 'please check the data')
  
  for _,kval in pairs(kvals) do
    print(string.format('%6.4f', correct[kval]/n*100.0))
  end
end

ExtractImg(opt.split, {num_images = opt.num_images})
Retrieval(opt.split, {num_images = opt.num_images})