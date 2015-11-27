--[[

  Concat layer: concatenates projected image features and word embeddings together
  after projecting image features to some dimension

--]]

local ConcatLayer, parent = torch.class('imagelstm.ConcatLayer', 'imagelstm.InputLayer')

function ConcatLayer:__init(config)
   parent.__init(self, config)
   self.emb = nn.LookupTable(self.vocab_size, self.emb_dim)
   
   -- Copy embedding weights
   if config.emb_vecs ~= nil then
     self.emb.weight:copy(config.emb_vecs)
   end
   -- image feature embedding
   self.combine_model = nn.Sequential()
                    :add(imagelstm.CRowJoinTable(2))

   if self.dropout then
    self.combine_model:add(nn.Dropout(self.dropout_prob, false))
   end

   self.params, self.grad_params = self.emb:getParameters()

  if gpu_mode then
    self:set_gpu_mode()
  end
end

-- Returns all of the weights of this module
function ConcatLayer:getWeights()
  return self.params
end

-- Sets gpu mode
function ConcatLayer:set_gpu_mode()
  self.combine_model:cuda()
  self.emb:cuda()
end

-- Sets cpu mode
function ConcatLayer:set_gpu_mode()
  self.combine_model:double()
  self.emb:double()
end

-- Enable Dropouts
function ConcatLayer:enable_dropouts()
   enable_sequential_dropouts(self.combine_model)
end

-- Disable Dropouts
function ConcatLayer:disable_dropouts()
   disable_sequential_dropouts(self.combine_model)
end
-- Does a single forward step of concat layer, concatenating
-- Input 
function ConcatLayer:forward(word_indeces, image_feats)
   parent:forward(word_indeces, image_feats, self.gpu_mode)

   self.word_proj = self.emb:forward(word_indeces)
   res = self.combine_model:forward({self.word_proj, image_feats})
   return res
end

function ConcatLayer:backward(word_indices, image_feats, err)
   parent:backward(word_indeces, image_feats, err, self.gpu_mode)

   emb_errors = self.combine_model:backward({self.word_proj, image_feats}, err)

   -- get the image and word projection errors
   image_emb_errors = emb_errors[2]
   word_proj_errors = emb_errors[1]

   self.emb:backward(word_indices, word_proj_errors)
end

-- Returns size of outputs of this combine module
function ConcatLayer:getOutputSize()
  return self.emb_dim + self.image_dim
end

function ConcatLayer:getParameters()
  return self.params, self.grad_params
end

-- zeros out the gradients
function ConcatLayer:zeroGradParameters() 
  self.emb:zeroGradParameters()
  self.combine_model:zeroGradParameters()
end

function ConcatLayer:getModules() 
  return {self.emb}
end

function ConcatLayer:normalizeGrads(batch_size)
  self.emb.gradWeight:div(batch_size)
end

