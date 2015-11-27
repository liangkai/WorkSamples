--[[

  Hidden Project Layer: Projects image input into projection dimension twice. For feeding in
  image input into lstm
--]]

local HiddenProjLayer, parent = torch.class('imagelstm.HiddenSingleProjLayer', 'imagelstm.HiddenLayer')

function HiddenProjLayer:__init(config)
   parent.__init(self, config)
   
   local modules = nn.Parallel()
   self.cell_activations = self.gpu_mode and torch.zeros(self.proj_dim):cuda()
                            or torch.zeros(self.proj_dim)
   -- image feature embedding
   if self.num_layers == 1 then 
    local hidden_image_emb = self:new_hidden_module()
    self.hidden_image_emb = hidden_image_emb
    modules:add(self.hidden_image_emb)
   else
    self.hidden_image_emb = {}
    for i = 1, self.num_layers do
      local hidden_image_emb = self:new_hidden_module()
      table.insert(self.hidden_image_emb, hidden_image_emb)
      modules:add(self.hidden_image_emb[i])
    end
   end
   
   self.params, self.grad_params = modules:getParameters()

  if gpu_mode then
    self:set_gpu_mode()
  end
end

function HiddenProjLayer:new_hidden_module() 
  local hidden_image_emb = nn.Sequential() 
        :add(nn.Linear(self.image_dim, self.proj_dim))

  if self.dropout then
      hidden_image_emb:add(nn.Dropout(self.dropout_prob, false))
  end
  return hidden_image_emb
end

-- Returns all of the weights of this module
function HiddenProjLayer:getWeights()
  return self.params
end

function HiddenProjLayer:getModules() 
  if self.num_layers == 1 then 
    return {self.hidden_image_emb}
  else 
    local modules = {}
    for i = 1, self.num_layers do
      table.insert(modules, self.hidden_image_emb[i])
    end
    return modules
  end
end

-- Sets gpu mode
function HiddenProjLayer:set_gpu_mode()
  if self.num_layers == 1 then 
     self.hidden_image_emb:cuda()
  else 
    for i = 1, self.num_layers do
       self.hidden_image_emb[i]:cuda()
    end
  end
end

function HiddenProjLayer:set_cpu_mode()
  if self.num_layers == 1 then 
     self.hidden_image_emb:double()
  else 
    for i = 1, self.num_layers do
       self.hidden_image_emb[i]:double()
    end
  end
end

-- Enable Dropouts
function HiddenProjLayer:enable_dropouts()
  if self.num_layers == 1 then 
    enable_sequential_dropouts(self.hidden_image_emb)
  else 
    for i = 1, self.num_layers do
      enable_sequential_dropouts(self.hidden_image_emb[i])
    end
  end
end

-- Disable Dropouts
function HiddenProjLayer:disable_dropouts()
  if self.num_layers == 1 then 
    disable_sequential_dropouts(self.hidden_image_emb)
  else 
    for i = 1, self.num_layers do
      disable_sequential_dropouts(self.hidden_image_emb[i])
    end
  end
end

-- Does a single forward step of concat layer, concatenating
-- 
function HiddenProjLayer:forward(image_feats)
   assert(image_feats ~= nil)
   assert(image_feats:size(1) == self.image_dim)
   parent:forward(image_feats, self.gpu_mode)

   if self.num_layers == 1 then
     self.cell_image_proj = self.cell_activations
     self.hidden_image_proj = self.hidden_image_emb:forward(image_feats)
     return {self.cell_image_proj, self.hidden_image_proj}
   else
     local cell_vals = {}
     local hidden_vals = {}

     for i = 1, self.num_layers do
      local cell_image_proj = self.cell_activations
      local hidden_image_proj = self.hidden_image_emb[i]:forward(image_feats)

      table.insert(cell_vals, cell_image_proj)
      table.insert(hidden_vals, hidden_image_proj)
     end

     return {cell_vals, hidden_vals}
   end
   
end

-- Does a single backward step of project layer
-- image_feats: input into hidden projection error
-- cell_errors: error of all hidden, cell units of lstm with respect to input
function HiddenProjLayer:backward(image_feats, cell_errors)
   assert(image_feats ~= nil)
   assert(image_feats:size(1) == self.image_dim)
   assert(cell_errors ~= nil)
   parent:backward(image_feats, cell_errors, self.gpu_mode)
   
   if self.num_layers == 1 then
     -- get the image and word projection errors
     local hidden_image_emb_errors = cell_errors[2]

     assert(hidden_image_emb_errors:size(1) == self.proj_dim)

     -- feed them backward
     self.hidden_image_emb:backward(image_feats, hidden_image_emb_errors)
   else
     for i = 1, self.num_layers do
        -- get the image and word projection errors
       local hidden_image_emb_errors = cell_errors[2][i]

       assert(hidden_image_emb_errors:size(1) == self.proj_dim)
       -- feed them backward
       self.hidden_image_emb[i]:backward(image_feats, hidden_image_emb_errors)
     end
   end
end

-- Returns size of outputs of this combine module
function HiddenProjLayer:getOutputSize()
  return self.mem_dim * 2 * self.num_layers
end

function HiddenProjLayer:getParameters()
  return self.params, self.grad_params
end

-- zeros out the gradients
function HiddenProjLayer:zeroGradParameters() 
  if self.num_layers == 1 then
    self.hidden_image_emb:zeroGradParameters()
  else
    for i = 1, self.num_layers do
      self.hidden_image_emb[i]:zeroGradParameters()
    end
  end
end

function HiddenProjLayer:normalizeGrads(batch_size)
  assert(batch_size ~= nil)
  if self.num_layers == 1 then
    self.hidden_image_emb.gradWeight:div(batch_size)
  else
    for i = 1, self.num_layers do
      self.hidden_image_emb[i].gradWeight:div(batch_size)
    end
  end
end

