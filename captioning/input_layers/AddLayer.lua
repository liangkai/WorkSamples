--[[

  Add layer: adds image features and word embeddings together
  at each step
--]]

local AddLayer, parent = torch.class('imagelstm.AddLayer', 'imagelstm.InputLayer')

function AddLayer:__init(config)
   parent.__init(self, config)

   -- word embeddings
   self.emb = nn.LookupTable(self.vocab_size, self.emb_dim)

   -- image feature embedding
   self.image_emb = nn.Linear(self.image_dim, self.emb_dim)

   -- Do a linear combination of image and word features
   local x1 = nn.Identity(self.emb_dim)()
   local x2 = nn.Identity(self.emb_dim)()
   local a = imagelstm.CRowAddTable()({x1, x2})
  
   self.lstm_emb = nn.Sequential()
              :add(nn.gModule({x1, x2}, {a}))
   if self.dropout then
      self.lstm_emb:add(nn.Dropout(self.dropout_prob, false))
  end

   local modules = nn.Parallel()
    :add(self.image_emb)
    :add(self.emb)

   self.params, self.grad_params = modules:getParameters()

   if self.gpu_mode then 
    self:set_gpu_mode()
   end

   if config.emb_vecs ~= nil then
    self.emb.weight:copy(config.emb_vecs)
   end

   -- Copy the image embedding vectors
   if config.combine_weights ~= nil then
     print("Copying combine weights")
     self.params:copy(config.combine_weights)
   end

end

-- Sets gpu mode
function AddLayer:set_gpu_mode()
  self.image_emb:cuda()
  self.emb:cuda()
  self.lstm_emb:cuda()
end

-- Returns all of the weights of this module
function AddLayer:getWeights()
  return self.params
end

function AddLayer:getModules() 
  return {self.image_emb, self.emb}
end

function AddLayer:set_cpu_mode()
  self.emb:double()
  self.lstm_emb:double()
  self.image_emb:double()
end

-- Enable Dropouts
function AddLayer:enable_dropouts()
   enable_sequential_dropouts(self.lstm_emb)
end

-- Disable Dropouts
function AddLayer:disable_dropouts()
   disable_sequential_dropouts(self.lstm_emb)
end


-- Does a single forward step of add layer
-- Word indeces: input tensor of word indeces
-- image_feats: Image features tensor
function AddLayer:forward(word_indeces, image_feats)
  parent:forward(word_indeces, image_feats, self.gpu_mode)

  self.text_inputs = self.emb:forward(word_indeces)
  self.image_inputs = self.image_emb:forward(image_feats)
  self.inputs = self.lstm_emb:forward({self.text_inputs, self.image_inputs})

    return self.inputs
end

function AddLayer:backward(word_indices, image_feats, grads)
  parent:backward(word_indeces, image_feats, grads, self.gpu_mode)
  -- backprop the gradients through the linear combination step
  local input_emb_grads = self.lstm_emb:backward({self.text_inputs, self.image_inputs}, grads)
  local emb_grads = input_emb_grads[1]
  local image_grads = input_emb_grads[2]

  self.emb:backward(word_indices, emb_grads)
  self.image_emb:backward(image_feats, image_grads)
end

function AddLayer:getOutputSize()
  return self.emb_dim
end

function AddLayer:getParameters()
  return self.params, self.grad_params
end

-- zeros out the gradients
function AddLayer:zeroGradParameters() 
  self.image_emb:zeroGradParameters()
  self.emb:zeroGradParameters()
  self.lstm_emb:zeroGradParameters()
end

function AddLayer:normalizeGrads(batch_size)
  self.image_emb.gradWeight:div(batch_size)
  self.emb.gradWeight:div(batch_size)
end

