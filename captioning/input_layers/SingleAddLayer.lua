--[[

  Add layer: adds image features and word embeddings together
  at first time step, then just word embeddings
--]]

local SingleAddLayer, parent = torch.class('imagelstm.SingleAddLayer', 'imagelstm.InputLayer')

function SingleAddLayer:__init(config)
   parent.__init(self, config)
   self.emb = nn.LookupTable(self.vocab_size, self.emb_dim)

   -- image feature embedding
   self.image_emb = nn.Linear(self.image_dim, self.emb_dim)

   -- Do a linear combination of image and word features
   local x1 = nn.Identity(self.emb_dim)()
   local x2 = nn.Identity(self.emb_dim)()
   local a = imagelstm.CRowSingleTable()({x1, x2})
  
   self.lstm_emb = nn.Sequential():
                add(nn.gModule({x1, x2}, {a}))

   if self.dropout then
    self.lstm_emb:add(nn.Dropout(self.dropout_prob, false))
   end

   if config.emb_vecs ~= nil then
     self.emb.weight:copy(config.emb_vecs)
   end

   local modules = nn.Parallel()
    :add(self.image_emb)
    :add(self.emb)

   self.params, self.grad_params = modules:getParameters()

   if self.gpu_mode then 
    self:set_gpu_mode()
   end
end

-- Sets gpu mode
function SingleAddLayer:set_gpu_mode()
  self.image_emb:cuda()
  self.emb:cuda()
  self.lstm_emb:cuda()
  self.params:cuda()
end

-- Sets cpu mode
function SingleAddLayer:set_cpu_mode()
  self.image_emb:double()
  self.emb:double()
  self.lstm_emb:double()
  self.params:double()
end

-- Enable Dropouts
function SingleAddLayer:enable_dropouts()
   enable_sequential_dropouts(self.lstm_emb)
end

-- Disable Dropouts
function SingleAddLayer:disable_dropouts()
   disable_sequential_dropouts(self.lstm_emb)
end

-- Returns the trainable modules of this layer
function SingleAddLayer:getModules() 
  return {self.image_emb, self.emb}
end

-- Returns all of the weights of this module
function SingleAddLayer:getWeights()
  return self.params
end

-- Does a single forward step of add layer
-- Num_iter: only for beam search since we forward image features on first index
function SingleAddLayer:forward(word_indeces, image_feats, num_iter)
    parent:forward(word_indeces, image_feats, self.gpu_mode)
    self.text_inputs = self.emb:forward(word_indeces)
   

    if num_iter ~= nil and num_iter > 0 then
      return self.text_inputs
    else
      self.image_inputs = self.image_emb:forward(image_feats)
      self.inputs = self.lstm_emb:forward({self.text_inputs, self.image_inputs})
      return self.inputs
    end
end

function SingleAddLayer:backward(word_indices, image_feats, grads)
  parent:backward(word_indices, image_feats, grads, self.gpu_mode)
  -- backprop the gradients through the linear combination step
  local input_emb_grads = self.lstm_emb:backward({self.text_inputs, self.image_inputs}, grads)
  local emb_grads = input_emb_grads[1]
  local image_grads = input_emb_grads[2]

  self.emb:backward(word_indices, emb_grads)
  self.image_emb:backward(image_feats, image_grads)
end

-- zeros out the gradients
function SingleAddLayer:zeroGradParameters() 
  self.image_emb:zeroGradParameters()
  self.emb:zeroGradParameters()
  self.lstm_emb:zeroGradParameters()
end

function SingleAddLayer:normalizeGrads(batch_size)
  self.image_emb.gradWeight:div(batch_size)
  self.emb.gradWeight:div(batch_size)
end

function SingleAddLayer:getOutputSize()
  return self.emb_dim
end

function SingleAddLayer:getParameters()
  return self.params, self.grad_params
end



