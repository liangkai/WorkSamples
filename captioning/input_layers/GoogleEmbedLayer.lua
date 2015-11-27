--[[

  GoogleEmbedLayer: Embedding layer for input into lstm, where at time step 0 image embedding
  is fed into LSTM. At time steps > 0 word embeddings are fed into the lstm

--]]

local GoogleEmbedLayer, parent = torch.class('imagelstm.GoogleEmbedLayer', 'imagelstm.InputLayer')

function GoogleEmbedLayer:__init(config)
  parent.__init(self, config)
  self.emb_table = nn.LookupTable(self.vocab_size, self.emb_dim)
    -- Copy embedding weights
  if config.emb_vecs ~= nil then
    self.emb_table.weight:copy(config.emb_vecs)
  end

  -- word embeddings
  self.emb = nn.Sequential()
            :add(self.emb_table)

  -- image feature embedding
  self.image_emb = nn.Sequential()
                      :add(nn.Linear(self.image_dim, self.emb_dim))

  if self.dropout then
    self.emb:add(nn.Dropout(self.dropout_prob, false))
    self.image_emb:add(nn.Dropout(self.dropout_prob, false))
  end

  local modules = nn.Parallel()
                    :add(self.emb)
                    :add(self.image_emb)

  self.params, self.grad_params = modules:getParameters()

  if self.gpu_mode then
    self:set_gpu_mode()
  end
end

-- Returns all of the weights of this module
function GoogleEmbedLayer:getWeights()
  return self.params
end

-- Sets gpu mode
function GoogleEmbedLayer:set_gpu_mode()
  self.emb:cuda()
  self.image_emb:cuda()
end

function GoogleEmbedLayer:set_cpu_mode()
  self.emb:double()
  self.image_emb:double()
end

-- Enable Dropouts
function GoogleEmbedLayer:enable_dropouts()
   enable_sequential_dropouts(self.emb)
   enable_sequential_dropouts(self.image_emb)
end

-- Disable Dropouts
function GoogleEmbedLayer:disable_dropouts()
   disable_sequential_dropouts(self.emb)
   disable_sequential_dropouts(self.image_emb)
end


-- Does a single forward step of concat layer, concatenating
-- word_indices, index of words
function GoogleEmbedLayer:forward(word_indeces, image_feats, num_iter)
   parent:forward(word_indeces, image_feats, self.gpu_mode)
   if num_iter ~= nil then
      if num_iter == 0 then 
         --print("Forwarding image")
         self.image_proj = self.image_emb:forward(image_feats)
         return {self.image_proj}
      else 
         --print("Forwarding word embeddings")
         self.word_proj = self.emb:forward(word_indeces)
         return self.word_proj
      end
   end

   self.image_proj = self.image_emb:forward(image_feats)
   self.word_proj = self.emb:forward(word_indeces)

   -- get number of tokens/time steps
   local T = word_indeces:size(1)
   local out = self.gpu_mode and torch.CudaTensor(T + 1, self.emb_dim):zero()
                or torch.Tensor(T + 1, self.emb_dim):zero()
   out[1] = self.image_proj
   local word_embs = out:narrow(1, 2, T)
   word_embs:copy(self.word_proj)

   -- TODO, concatenate them
   return out
end

function GoogleEmbedLayer:backward(word_indices, image_feats, err)
   parent:backward(word_indices, image_feats, err, self.gpu_mode)

   local T = err:size(1) - 1
   local image_err = err[1]
   local emb_err = err:narrow(1, 2, T)

   self.image_emb:backward(image_feats, image_err)
   self.emb:backward(word_indices, emb_err)
end

-- Returns size of outputs of this combine module
function GoogleEmbedLayer:getOutputSize()
  return self.emb_dim
end

function GoogleEmbedLayer:getParameters()
  return self.params, self.grad_params
end

-- zeros out the gradients
function GoogleEmbedLayer:zeroGradParameters() 
  self.emb:zeroGradParameters()
end

function GoogleEmbedLayer:getModules() 
  return {self.emb, self.image_emb}
end

function GoogleEmbedLayer:normalizeGrads(batch_size)
  self.emb.gradWeight:div(batch_size)
end

