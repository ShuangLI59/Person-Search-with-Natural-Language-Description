require 'nn'
require 'nngraph'

local Attention = {}
function Attention.attn(h_size, emb_size, out_size)  
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- h(t)
  table.insert(inputs, nn.Identity()()) -- img 

  -- generate visual units
  local fc1 = nn.Linear(4096, emb_size)(inputs[2])
  local bn = nn.BatchNormalization(emb_size)(fc1)
  local relu = nn.ReLU(true)(bn)
  local fc2 = nn.Linear(emb_size, out_size)(relu)

  -- word-level gate
  local gate_fc = nn.Linear(h_size, 1)(inputs[1])
  local word_gate = nn.Sigmoid()(gate_fc)

  -- Unit-level attention
  local attention_fc = nn.Linear(h_size, out_size)(inputs[1])
  local unit_attention = nn.SoftMax()(attention_fc)

  -- Word-image affinity
  local dotproct = nn.DotProduct()({unit_attention, fc2})
  local unsqueeze = nn.View(-1, 1)(dotproct)
  local affinity = nn.CMulTable()({unsqueeze, word_gate})

  local outputs = {}
  table.insert(outputs, affinity)

  return nn.gModule(inputs, outputs)
end

return Attention

