--[[

  Embed layer: Simple word embedding layer for input into lstm

--]]

local EmbedLayer, parent = torch.class('imagelstm.EmbedLayer', 'imagelstm.InputLayer')

function EmbedLayer:__init(config)
  parent.__init(self, config)
  self.emb_table = nn.LookupTable(self.vocab_size, self.emb_dim)
    -- Copy embedding weights
  if config.emb_vecs ~= nil then
    print("Initializing embeddings from config ")
    self.emb_table.weight:copy(config.emb_vecs)
  end

  self.emb = nn.Sequential()
            :add(self.emb_table)

  if self.dropout then
    self.emb:add(nn.Dropout(self.dropout_prob, false))
  end


  self.params, self.grad_params = self.emb:getParameters()

  if self.gpu_mode then
    self:set_gpu_mode()
  end
end

-- Returns all of the weights of this module
function EmbedLayer:getWeights()
  return self.params
end

-- Sets gpu mode
function EmbedLayer:set_gpu_mode()
  self.emb:cuda()
  self.gpu_mode = true
end

function EmbedLayer:set_cpu_mode()
  self.emb:double()
  self.gpu_mode = false
end

-- Enable Dropouts
function EmbedLayer:enable_dropouts()
   enable_sequential_dropouts(self.emb)
end

-- Disable Dropouts
function EmbedLayer:disable_dropouts()
   disable_sequential_dropouts(self.emb)
end


-- Does a single forward step of concat layer, concatenating
-- Input 
function EmbedLayer:forward(word_indeces, image_feats)
   parent:forward(word_indeces, image_feats, self.gpu_mode)
   self.word_proj = self.emb:forward(word_indeces)
   return self.word_proj
end

function EmbedLayer:backward(word_indices, image_feats, err)
   parent:backward(word_indices, image_feats, err, self.gpu_mode)
   self.emb:backward(word_indices, err)
end

-- Returns size of outputs of this combine module
function EmbedLayer:getOutputSize()
  return self.emb_dim
end

function EmbedLayer:getParameters()
  return self.params, self.grad_params
end

-- zeros out the gradients
function EmbedLayer:zeroGradParameters() 
  self.emb:zeroGradParameters()
end

function EmbedLayer:getModules() 
  return {self.emb}
end

function EmbedLayer:normalizeGrads(batch_size)
  self.emb.gradWeight:div(batch_size)
end

