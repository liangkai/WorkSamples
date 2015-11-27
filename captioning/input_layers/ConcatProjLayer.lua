--[[

  Concat layer: concatenates projected image features and word embeddings together
  after projecting image features to some dimension

--]]

local ConcatProjLayer, parent = torch.class('imagelstm.ConcatProjLayer', 'imagelstm.InputLayer')

function ConcatProjLayer:__init(config)
   parent.__init(self, config)
   self.emb = nn.LookupTable(self.vocab_size, self.emb_dim)
   -- image feature embedding
   self.image_emb = nn.Linear(self.image_dim, self.emb_dim)
   self.combine_model = nn.Sequential()
                      :add(imagelstm.CRowJoinTable(2))
   if self.dropout then
     self.combine_model:add(nn.Dropout(self.dropout_prob, false))
   end

   if config.emb_vecs ~= nil then
     self.emb.weight:copy(config.emb_vecs)
   end

   local modules = nn.Parallel()
    :add(self.image_emb)
    :add(self.emb)

   self.params, self.grad_params = modules:getParameters()

   -- Copy embedding vectors
   if config.emb_vecs ~= nil then
     self.emb.weight:copy(config.emb_vecs)
   end

   -- Copy the image embedding vectors
   if config.combine_weights ~= nil then
     self.params:copy(config.combine_weights)
   end

  if self.gpu_mode then
    self:set_gpu_mode()
  end
end

-- Returns all of the weights of this module
function ConcatProjLayer:getWeights()
  return self.params
end

function ConcatProjLayer:getModules() 
  return {self.emb, self.image_emb}
end

-- Sets gpu mode
function ConcatProjLayer:set_gpu_mode()
  self.image_emb:cuda()
  self.combine_model:cuda()
  self.emb:cuda()
end

-- Sets cpu mode
function ConcatProjLayer:set_cpu_mode()
  self.image_emb:double()
  self.combine_model:double()
  self.emb:double()
end

-- Enable Dropouts
function ConcatProjLayer:enable_dropouts()
   enable_sequential_dropouts(self.combine_model)
end

-- Disable Dropouts
function ConcatProjLayer:disable_dropouts()
   disable_sequential_dropouts(self.combine_model)
end

-- Does a single forward step of concat layer, concatenating
-- Input 
function ConcatProjLayer:forward(word_indices, image_feats)
   -- parent does checks
   parent:forward(word_indices, image_feats, self.gpu_mode)

   self.image_proj = self.image_emb:forward(image_feats)
   self.word_proj = self.emb:forward(word_indices)
   res = self.combine_model:forward({self.word_proj, self.image_proj})
   return res
end

function ConcatProjLayer:backward(word_indices, image_feats, err)
   parent:backward(word_indices, image_feats, err, self.gpu_mode)

   emb_errors = self.combine_model:backward({self.word_proj, self.image_proj}, err)

   -- get the image and word projection errors
   image_emb_errors = emb_errors[2]
   word_proj_errors = emb_errors[1]

   -- feed them backward
   self.image_emb:backward(image_feats, image_emb_errors)
   self.emb:backward(word_indices, word_proj_errors)
end

-- Returns size of outputs of this combine module
function ConcatProjLayer:getOutputSize()
  return self.emb_dim + self.emb_dim
end

function ConcatProjLayer:getParameters()
  return self.params, self.grad_params
end

function ConcatProjLayer:getModules() 
  return {self.emb, self.image_emb}
end

-- zeros out the gradients
function ConcatProjLayer:zeroGradParameters() 
  self.emb:zeroGradParameters()
  self.image_emb:zeroGradParameters()
  self.combine_model:zeroGradParameters()
end

function ConcatProjLayer:normalizeGrads(batch_size)
  self.emb.gradWeight:div(batch_size)
  self.image_emb.gradWeight:div(batch_size)
end
