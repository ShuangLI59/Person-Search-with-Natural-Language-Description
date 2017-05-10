require 'torch'
require 'nn'
require 'nngraph'
require 'loadcaffe'
require 'gnuplot'
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'
require 'misc.DataLoader'
require 'misc.optim_updates'
require 'misc.LanguageModel'
-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Person Search with Natural Language Description')
cmd:text()
cmd:text('Options')

-- Data input settings
cmd:option('-input_h5','data/reidtalk.h5','path to the h5file containing the preprocessed dataset')
cmd:option('-input_json','data/reidtalk.json','path to the json file containing additional info and vocab')
cmd:option('-cnn_proto','model/VGG_ILSVRC_16_layers_deploy.prototxt','path to CNN prototxt file in Caffe format.')
cmd:option('-cnn_model','model/VGG16_iter_50000.caffemodel','path to VGG-16 Visual CNN')
cmd:option('-start_from', '', 'path to a model checkpoint to initialize model weights from. Empty = don\'t')

-- Model settings
cmd:option('-neg_time',3)
cmd:option('-input_encoding_size',512,'the encoding size of each token in the vocabulary, and the image.')
cmd:option('-rnn_size',512,'size of the rnn in number of hidden nodes in each layer')
-- Optimization: General
cmd:option('-batch_size',32)
cmd:option('-grad_clip',5,'clip gradients at this value (note should be lower than usual 5 because we normalize grads by both batch and seq_length)')
cmd:option('-drop_prob_lm', 0.5, 'strength of dropout in the Language Model RNN')
cmd:option('-finetune_cnn_after', -1, 'After what iteration do we start finetuning the CNN? (-1 = disable; never finetune, 0 = finetune from start)')
cmd:option('-max_iters', -1, 'max number of iterations to run for (-1 = run forever)')
-- cmd:option('-seq_per_img',1,'number of captions to sample for each image during training.')
-- Optimization: for the Language Model
cmd:option('-optim','adam','what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
cmd:option('-learning_rate',0.0004,'learning rate')
cmd:option('-learning_rate_decay_start', -1, 'at what iteration to start decaying learning rate? (-1 = dont)')
cmd:option('-learning_rate_decay_every', 50000, 'every how many iterations thereafter to drop LR by half?')
cmd:option('-optim_alpha',0.8,'alpha for adagrad/rmsprop/momentum/adam')
cmd:option('-optim_beta',0.999,'beta used for adam')
cmd:option('-optim_epsilon',1e-8,'epsilon that goes into denominator for smoothing')
cmd:option('-num_layers',1,'number of LSTM layers')
-- Optimization: for the CNN
cmd:option('-cnn_optim','adam','optimization to use for CNN')
cmd:option('-cnn_learning_rate',1e-5,'learning rate for the CNN')
cmd:option('-cnn_weight_decay', 0, 'L2 weight decay just for the CNN')
cmd:option('-cnn_optim_alpha',0.8,'alpha for momentum of CNN')
cmd:option('-cnn_optim_beta',0.999,'alpha for momentum of CNN')

-- Evaluation/Checkpointing
cmd:option('-val_images_use', 500, 'how many images to use when periodically evaluating the validation loss? (-1 = all)')
cmd:option('-save_checkpoint_every', 500, 'how often to save a model checkpoint?')
cmd:option('-checkpoint_path', 'snapshot', 'folder to save checkpoints into (empty = this folder)')
cmd:option('-losses_log_every', 25, 'How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')

-- misc
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-id', '', 'an id identifying this run/job. used in cross-val and appended when writing progress files')
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
print(opt)

-------------------------------------------------------------------------------
-- Create the Data Loader instance
-------------------------------------------------------------------------------
local loader = DataLoader{h5_file = opt.input_h5, json_file = opt.input_json}
local nTrain = (#loader.split_ix['train'])

-------------------------------------------------------------------------------
-- Initialize the networks
-------------------------------------------------------------------------------
local protos = {}
if string.len(opt.start_from) > 0 then
  -- load protos from file
  print('initializing weights from ' .. opt.start_from)
  local loaded_checkpoint = torch.load(opt.start_from)
  protos = loaded_checkpoint.protos
  net_utils.unsanitize_gradients(protos.cnn)
  local lm_modules = protos.lm:getModulesList()
  for k,v in pairs(lm_modules) do net_utils.unsanitize_gradients(v) end
  protos.crit = nn.BCECriterion() -- not in checkpoints, create manually
else
  -- create protos from scratch,  intialize language model
  local lmOpt = {}
  lmOpt.vocab_size = loader:getVocabSize()
  lmOpt.input_encoding_size = opt.input_encoding_size
  lmOpt.rnn_size = opt.rnn_size
  lmOpt.num_layers = opt.num_layers
  lmOpt.dropout = opt.drop_prob_lm
  lmOpt.seq_length = loader:getSeqLength()
  lmOpt.batch_size = opt.batch_size
  protos.lm = nn.LanguageModel(lmOpt)
  -- initialize the ConvNet
  local cnn_backend = opt.backend
  if opt.gpuid == -1 then cnn_backend = 'nn' end -- override to nn if gpu is disabled
  
  local cnn_raw = loadcaffe.load(opt.cnn_proto, opt.cnn_model, cnn_backend)
  protos.cnn = net_utils.build_cnn(cnn_raw, {encoding_size = opt.input_encoding_size, backend = cnn_backend})

  protos.crit = nn.BCECriterion()
end


-- ship everything to GPU, maybe
if opt.gpuid >= 0 then
  for k,v in pairs(protos) do v:cuda() end
end

-- flatten and prepare all model parameters to a single vector. 
-- Keep CNN params separate in case we want to try to get fancy with different optims on LM/CNN
local params, grad_params = protos.lm:getParameters()
local cnn_params, cnn_grad_params = protos.cnn:getParameters()
print('total number of parameters in LM: ', params:nElement())
print('total number of parameters in CNN: ', cnn_params:nElement())
assert(params:nElement() == grad_params:nElement())
assert(cnn_params:nElement() == cnn_grad_params:nElement())

-- construct thin module clones that share parameters with the actual
-- modules. These thin module will have no intermediates and will be used
-- for checkpointing to write significantly smaller checkpoint files
local thin_lm = protos.lm:clone()
thin_lm.lookup_table:share(protos.lm.lookup_table, 'weight', 'bias')
thin_lm.emb_img:share(protos.lm.emb_img, 'weight', 'bias', 'running_mean', 'running_var')
thin_lm.core:share(protos.lm.core, 'weight', 'bias')
thin_lm.attention:share(protos.lm.attention, 'weight', 'bias', 'running_mean', 'running_var')
thin_lm.sigmoid:share(protos.lm.sigmoid, 'weight', 'bias')

local thin_cnn = protos.cnn:clone('weight', 'bias')
net_utils.sanitize_gradients(thin_cnn)
local lm_modules = thin_lm:getModulesList()
for k,v in pairs(lm_modules) do net_utils.sanitize_gradients(v) end

-- create clones and ensure parameter sharing. we have to do this 
-- all the way here at the end because calls such as :cuda() and
-- :getParameters() reshuffle memory around.
protos.lm:createClones()

collectgarbage()


-------------------------------------------------------------------------------
-- Validation evaluation
-------------------------------------------------------------------------------
local function eval_split(split, evalopt)
  local verbose = utils.getopt(evalopt, 'verbose', true)
  local val_images_use = utils.getopt(evalopt, 'val_images_use', true)

  protos.cnn:evaluate()
  protos.lm:evaluate()
  loader:resetIterator(split) -- rewind iteator back to first datapoint in the split
  local n = 0
  local loss_sum = 0
  local loss_evals = 0
  top1_val = 0
  local vocab = loader:getVocab()
  while true do

    -- fetch a batch of data
    local data = loader:getBatch{batch_size = opt.batch_size, split = split, seq_per_img = 1, neg_time = opt.neg_time}
    data.images = net_utils.prepro(data.images, false, opt.gpuid >= 0) -- preprocess in place, and don't augment
    n = n + opt.batch_size

    -- forward the model to get loss
    local feats = protos.cnn:forward(data.images)
    local logprobs = protos.lm:forward{feats, data.labels, data.seqlen}
    local loss = protos.crit:forward(logprobs, data.cls:cuda())

    loss_sum = loss_sum + loss
    loss_evals = loss_evals + 1

    -- compute top1 accuracy
    local predictions = logprobs:float():ge(0.5)
    local correct = predictions:long():eq(data.cls:long():view(logprobs:size(1), 1):expandAs(logprobs))
    top1_val = top1_val + correct:sum() / logprobs:size(1)
    -- if we wrapped around the split or used up val imgs budget then bail
    local ix0 = data.bounds.it_pos_now
    local ix1 = math.min(data.bounds.it_max, val_images_use)
    if verbose then
      print(string.format('evaluating validation performance... %d/%d (%f)', ix0-1, ix1, loss))
    end

    if loss_evals % 10 == 0 then collectgarbage() end
    if data.bounds.wrapped then break end -- the split ran out of data, lets break out
    if n >= val_images_use then break end -- we've used enough images
  end

  top1_val = top1_val/loss_evals
  return loss_sum/loss_evals
end

-------------------------------------------------------------------------------
-- Loss function
-------------------------------------------------------------------------------
local iter = 0
local function lossFun()
  protos.cnn:training()
  protos.lm:training()
  grad_params:zero()
  if opt.finetune_cnn_after >= 0 and iter >= opt.finetune_cnn_after then
    cnn_grad_params:zero()
  end

  -----------------------------------------------------------------------------
  -- Forward pass
  -----------------------------------------------------------------------------
  -- get batch of data  
  local data = loader:getBatch{batch_size = opt.batch_size, split = 'train', seq_per_img = 1, neg_time = opt.neg_time}
  data.images = net_utils.prepro(data.images, true, opt.gpuid >= 0) -- preprocess in place, do data augmentation

  -- forward the ConvNet on images (most work happens here)
  local feats = protos.cnn:forward(data.images)
  -- forward the language model
  local logprobs = protos.lm:forward{feats, data.labels, data.seqlen} 
  -- forward the language model criterion
  local loss = protos.crit:forward(logprobs, data.cls:cuda()) 

  -- compute top1 accuracy
  local predictions = logprobs:float():ge(0.5)
  local correct = predictions:long():eq(data.cls:long():view(logprobs:size(1), 1):expandAs(logprobs))
  top1 = correct:sum() / logprobs:size(1)
  ---------------------------------------------------------------------------
  -- Backward pass
  -----------------------------------------------------------------------------
  -- backprop criterion
  local dlogprobs = protos.crit:backward(logprobs, data.cls:cuda())
  -- backprop language model
  local dfeats, ddummy = unpack(protos.lm:backward({feats, data.labels, data.seqlen}, dlogprobs))
  -- backprop the CNN, but only if we are finetuning
  if opt.finetune_cnn_after >= 0 and iter >= opt.finetune_cnn_after then
    local dx = protos.cnn:backward(data.images, dfeats)
  end

  -- clip gradients
  grad_params:clamp(-opt.grad_clip, opt.grad_clip)

  -- apply L2 regularization
  if opt.cnn_weight_decay > 0 then
    cnn_grad_params:add(opt.cnn_weight_decay, cnn_params)
    cnn_grad_params:clamp(-opt.grad_clip, opt.grad_clip)
  end
  -----------------------------------------------------------------------------

  -- and lets get out!
  local losses = { total_loss = loss }
  return losses
end

-------------------------------------------------------------------------------
-- Main loop
-------------------------------------------------------------------------------
local loss0
local optim_state = {}
local cnn_optim_state = {}
local loss_history = {}
local val_loss_history = {}
local acc_history = {}
local val_acc_history = {}
local best_score_ACC
top1 = 0
top1_val = 0

while true do  
  local epoch = iter / (nTrain/opt.batch_size)

  ---- train loss
  local losses = lossFun()
  if iter % opt.losses_log_every == 0 then 
    loss_history[iter] = losses.total_loss 
    acc_history[iter] = top1
  end
  print(string.format('iter %d: loss: %f acc: %f', iter, losses.total_loss, top1))

  ---- save checkpoint once in a while (or on final iteration)
  if (iter % opt.save_checkpoint_every == 0 or iter == opt.max_iters) then

    ---- eval loss
    local val_loss = eval_split('val', {val_images_use = opt.val_images_use})
    print(string.format('validation loss: %f validation acc: %f', val_loss, top1_val))
    val_loss_history[iter] = val_loss
    val_acc_history[iter] = top1_val

    ---- save checkpoint
    local checkpoint_path = string.format('%s/lstm%s_rnn%s_epoch%.2f_valloss%.4f_valacc%.4f', opt.checkpoint_path, opt.num_layers, opt.rnn_size, epoch, val_loss, top1_val)
    local checkpoint_path_best_ACC = string.format('%s/lstm%s_rnn%s_bestACC', opt.checkpoint_path, opt.num_layers, opt.rnn_size)
    ---- write a (thin) json report
    local checkpoint = {}
    checkpoint.opt = opt
    checkpoint.iter = iter
    checkpoint.loss_history = loss_history
    checkpoint.val_loss_history = val_loss_history
    checkpoint.acc_history = acc_history
    checkpoint.val_acc_history = val_acc_history

    utils.write_json(checkpoint_path .. '.json', checkpoint)
    print('wrote json checkpoint to ' .. checkpoint_path .. '.json')

    -- write the full model checkpoint as well if we did better than ever
    local current_score = top1_val
    if best_score_ACC == nil or current_score > best_score_ACC then
      best_score_ACC = current_score
      if iter > 0 then -- dont save on very first iteration
        -- include the protos (which have weights) and save to file
        local save_protos = {}
        save_protos.lm = thin_lm -- these are shared clones, and point to correct param storage
        save_protos.cnn = thin_cnn
        checkpoint.protos = save_protos
        -- also include the vocabulary mapping so that we can use the checkpoint 
        -- alone to run on arbitrary images without the data loader
        checkpoint.vocab = loader:getVocab()
        torch.save(checkpoint_path_best_ACC .. '.t7', checkpoint)
        print('wrote checkpoint to ' .. checkpoint_path_best_ACC .. '.t7')
      end
    end

  end

  -- decay the learning rate for both LM and CNN
  local learning_rate = opt.learning_rate
  local cnn_learning_rate = opt.cnn_learning_rate
  if iter > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0 then
    local frac = (iter - opt.learning_rate_decay_start) / opt.learning_rate_decay_every
    local decay_factor = math.pow(0.5, frac)
    learning_rate = learning_rate * decay_factor -- set the decayed rate
    cnn_learning_rate = cnn_learning_rate * decay_factor
  end
  
  -- perform a parameter update
  if opt.optim == 'rmsprop' then
    rmsprop(params, grad_params, learning_rate, opt.optim_alpha, opt.optim_epsilon, optim_state)
  elseif opt.optim == 'adagrad' then
    adagrad(params, grad_params, learning_rate, opt.optim_epsilon, optim_state)
  elseif opt.optim == 'sgd' then
    sgd(params, grad_params, opt.learning_rate)
  elseif opt.optim == 'sgdm' then
    sgdm(params, grad_params, learning_rate, opt.optim_alpha, optim_state)
  elseif opt.optim == 'sgdmom' then
    sgdmom(params, grad_params, learning_rate, opt.optim_alpha, optim_state)
  elseif opt.optim == 'adam' then
    adam(params, grad_params, learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, optim_state)
  else
    error('bad option opt.optim')
  end

  -- do a cnn update (if finetuning, and if rnn above us is not warming up right now)
  if opt.finetune_cnn_after >= 0 and iter >= opt.finetune_cnn_after then
    if opt.cnn_optim == 'sgd' then
      sgd(cnn_params, cnn_grad_params, cnn_learning_rate)
    elseif opt.cnn_optim == 'sgdm' then
      sgdm(cnn_params, cnn_grad_params, cnn_learning_rate, opt.cnn_optim_alpha, cnn_optim_state)
    elseif opt.cnn_optim == 'adam' then
      adam(cnn_params, cnn_grad_params, cnn_learning_rate, opt.cnn_optim_alpha, opt.cnn_optim_beta, opt.optim_epsilon, cnn_optim_state)
    else
      error('bad option for opt.cnn_optim')
    end
  end

  -- stopping criterions
  iter = iter + 1
  if iter % 10 == 0 then collectgarbage() end -- good idea to do this once in a while, i think
  if loss0 == nil then loss0 = losses.total_loss end
  if losses.total_loss > loss0 * 20 then
    print('loss seems to be exploding, quitting.')
    break
  end
  if opt.max_iters > 0 and iter >= opt.max_iters then break end -- stopping criterion

end
