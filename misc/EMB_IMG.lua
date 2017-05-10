require 'nn'
require 'nngraph'

local EMB_IMG = {}
function EMB_IMG.emb(output_size)  
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- txt
  table.insert(inputs, nn.Identity()()) -- img

  local fc1 = nn.Linear(4096, output_size)(inputs[2])
  local bn = nn.BatchNormalization(output_size)(fc1)
  local relu = nn.ReLU(true)(bn)
  local fc2 = nn.Linear(output_size, output_size)(relu)


  local concatenation = nn.JoinTable(2)({inputs[1],fc2})

  local outputs = {}
  table.insert(outputs, concatenation)

  return nn.gModule(inputs, outputs)
end

return EMB_IMG

