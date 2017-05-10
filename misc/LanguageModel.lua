require 'nn'
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'
local LSTM = require 'misc.LSTM'
local Attention = require 'misc.Attention'
local EMB_IMG = require 'misc.EMB_IMG'
-------------------------------------------------------------------------------
-- Language Model core
-------------------------------------------------------------------------------
local layer, parent = torch.class('nn.LanguageModel', 'nn.Module')
function layer:__init(opt)
  parent.__init(self)
  -- options for core network
  self.vocab_size = utils.getopt(opt, 'vocab_size') -- required
  self.input_encoding_size = utils.getopt(opt, 'input_encoding_size')
  self.rnn_size = utils.getopt(opt, 'rnn_size')
  self.num_layers = utils.getopt(opt, 'num_layers', 1)
  local dropout = utils.getopt(opt, 'dropout', 0)
  -- options for Language Model
  self.seq_length = utils.getopt(opt, 'seq_length')
  self.emb_size = self.rnn_size 

  self.lookup_table = nn.LookupTable(self.vocab_size, self.input_encoding_size)
  self.emb_img = EMB_IMG.emb(self.input_encoding_size)
  self.core = LSTM.lstm(self.input_encoding_size*2, self.rnn_size, self.num_layers, dropout)
  self.attention = Attention.attn(self.rnn_size, self.emb_size, self.input_encoding_size)
  self.sigmoid = nn.Sigmoid()
  self:_createInitState(1) -- will be lazily resized later during forward passes
end

function layer:_createInitState(batch_size)
  assert(batch_size ~= nil, 'batch size must be provided')
  -- construct the initial state for the LSTM
  if not self.init_state then self.init_state = {} end -- lazy init
  for h=1,self.num_layers*2 do
    -- note, the init state Must be zeros because we are using init_state to init grads in backward call too
    if self.init_state[h] then
      if self.init_state[h]:size(1) ~= batch_size then
        self.init_state[h]:resize(batch_size, self.rnn_size):zero() -- expand the memory
      end
    else
      self.init_state[h] = torch.zeros(batch_size, self.rnn_size)
    end
  end
  self.num_state = #self.init_state
end

function layer:createClones()
  -- construct the net clones
  print('constructing clones inside the LanguageModel')
  self.lookup_tables = {self.lookup_table}
  self.emb_imgs = {self.emb_img}
  self.clones = {self.core}
  self.attentions = {self.attention}
  
  for t=2,self.seq_length do
    self.lookup_tables[t] = self.lookup_table:clone('weight', 'bias', 'gradWeight', 'gradBias')
    self.emb_imgs[t] = self.emb_img:clone('weight', 'bias', 'gradWeight', 'gradBias', 'running_mean', 'running_var')
    self.clones[t] = self.core:clone('weight', 'bias', 'gradWeight', 'gradBias')
    self.attentions[t] = self.attention:clone('weight', 'bias', 'gradWeight', 'gradBias', 'running_mean', 'running_var')
  end
end

function layer:getModulesList()
  return {self.lookup_table, self.emb_img, self.core, self.attention, self.sigmoid}
end

function layer:parameters()
  -- we only have two internal modules, return their params
  local p1,g1 = self.lookup_table:parameters()
  local p2,g2 = self.emb_img:parameters()
  local p3,g3 = self.core:parameters()
  local p4,g4 = self.attention:parameters()

  local params = {}
  for k,v in pairs(p1) do table.insert(params, v) end
  for k,v in pairs(p2) do table.insert(params, v) end
  for k,v in pairs(p3) do table.insert(params, v) end
  for k,v in pairs(p4) do table.insert(params, v) end
  
  local grad_params = {}
  for k,v in pairs(g1) do table.insert(grad_params, v) end
  for k,v in pairs(g2) do table.insert(grad_params, v) end
  for k,v in pairs(g3) do table.insert(grad_params, v) end
  for k,v in pairs(g4) do table.insert(grad_params, v) end

  return params, grad_params
end

function layer:training()
  if self.clones == nil then self:createClones() end -- create these lazily if needed
  for k,v in pairs(self.lookup_tables) do v:training() end
  for k,v in pairs(self.emb_imgs) do v:training() end
  for k,v in pairs(self.clones) do v:training() end
  for k,v in pairs(self.attentions) do v:training() end
  self.sigmoid:training()
end

function layer:evaluate()
  if self.clones == nil then self:createClones() end -- create these lazily if needed
  for k,v in pairs(self.lookup_tables) do v:evaluate() end
  for k,v in pairs(self.emb_imgs) do v:evaluate() end
  for k,v in pairs(self.clones) do v:evaluate() end
  for k,v in pairs(self.attentions) do v:evaluate() end
  self.sigmoid:evaluate()
end


function layer:updateOutput(input)
  local imgs = input[1]
  local seq = input[2] 
  local seqlen = input[3] 
  local batch_size = seq:size(2)
  assert(seq:size(1) == self.seq_length)

  self.mask = torch.CudaByteTensor() 
  self.mask:resize(seq:size()):zero()
  self.mask[torch.eq(seq, 0)] = 1  

  if self.clones == nil then self:createClones() end -- lazily create clones on first forward pass
  self.tmax = torch.max(seqlen)
  self.tmin = torch.min(seqlen)
  
  self.lookup_tables_inputs = {}
  self.emb_img_input = {}
  self.inputs = {}
  self.attentions_inputs = {}
  
  self.output:resize(self.seq_length, batch_size, self.rnn_size)
  self.outputAttn = torch.CudaTensor(self.seq_length, batch_size):zero()
  self.outlast = torch.CudaTensor(batch_size, 1):zero()
  self.outsigmoid = torch.CudaTensor(batch_size, 1):zero()

  self:_createInitState(batch_size)
  self.state = {[0] = self.init_state}  
  for t=1,self.tmax do

    local it = seq[t]:clone()
    it[torch.eq(it,0)] = 1
    self.lookup_tables_inputs[t] = it
    local xt = self.lookup_tables[t]:forward(self.lookup_tables_inputs[t])

    self.emb_img_input[t] = {xt, imgs}
    local emb_img_output = self.emb_imgs[t]:forward(self.emb_img_input[t])

    -- lstm
    self.inputs[t] = {emb_img_output,unpack(self.state[t-1])}
    local out = self.clones[t]:forward(self.inputs[t]) 

    if t>self.tmin then
      for i=1,self.num_state+1 do
        out[i]:maskedFill(self.mask[t]:view(batch_size,1):expandAs(out[i]), 0)
      end
    end

    self.output[t] = out[self.num_state+1] -- last element is the output vector

    self.state[t] = {} -- the rest is state
    for i=1,self.num_state do table.insert(self.state[t], out[i]) end

    -- attention model
    self.attentions_inputs[t] = {self.output[t], imgs}
    self.outputAttn[t] = self.attentions[t]:forward(self.attentions_inputs[t])
  end

  self.outputAttn:maskedFill(self.mask, 0)
  self.outlast = self.outputAttn:sum(1):view(-1,1)--:cdiv(seqlen:cuda())
  self.outsigmoid = self.sigmoid:forward(self.outlast)
  return self.outsigmoid
end



function layer:updateGradInput(input, gradOutput)  
  local imgs = input[1]
  local seq = input[2] 
  local seqlen = input[3] 
  local batch_size = seq:size(2)

  local dimgs
  local dsigmoid = self.sigmoid:backward(self.outlast, gradOutput)
  --local doutputAttn = dsigmoid:cdiv(seqlen:cuda()):repeatTensor(1, self.seq_length):transpose(1,2):contiguous()
  local doutputAttn = dsigmoid:repeatTensor(1, self.seq_length):transpose(1,2):contiguous()
  doutputAttn:maskedFill(self.mask, 0)

  -- go backwards and lets compute gradients
  local dstate = {[self.tmax] = self.init_state} -- this works when init_state is all zeros
  for t=self.tmax,1,-1 do
    local doutput
    doutput, dimgs = unpack(self.attentions[t]:backward(self.attentions_inputs[t], doutputAttn[t]))

    local dout = {}
    for k=1,#dstate[t] do table.insert(dout, dstate[t][k]) end
    table.insert(dout, doutput)

    if t>self.tmin then
      for i=1,self.num_state+1 do
        dout[i]:maskedFill(self.mask[t]:view(batch_size,1):expandAs(dout[i]), 0)
      end
    end

    local dinputs = self.clones[t]:backward(self.inputs[t], dout)


    dstate[t-1] = {}
    for k=2,self.num_state+1 do table.insert(dstate[t-1], dinputs[k]) end
    -------------------------------------------------------------------------
    -- compute grad_txt and grad_img seperately
    -------------------------------------------------------------------------
    local demb_imgs = self.emb_imgs[t]:backward(self.emb_img_input[t], dinputs[1])
    local dxt = demb_imgs[1]
    dimgs:add(demb_imgs[2])

    self.lookup_tables[t]:backward(self.lookup_tables_inputs[t], dxt)
  end

  -- we have gradient on image, but for LongTensor gt sequence we only create an empty tensor - can't backprop
  self.gradInput = {dimgs, torch.Tensor()}
  return self.gradInput
end
