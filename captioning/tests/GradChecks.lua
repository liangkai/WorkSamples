--[[

  Image Captioning using an LSTM and a softmax for the language model

--]]

local GradChecks = torch.class('imagelstm.GradChecks')

function GradChecks:__init(config)
  self.tol = 1e-5
  self.in_dim = 4
  self.mem_dim = 3
  self.num_classes = 5
  self.reg = 0.5
  self.criterion = nn.ClassNLLCriterion()
end

function GradChecks:new_caption_module()
  local caption_module = nn.Sequential()

  caption_module
    :add(nn.Linear(self.mem_dim, self.num_classes))
    --:add(nn.Dropout())
    :add(nn.LogSoftMax())
  return caption_module
end

function GradChecks:check_addlayer()
  input = torch.rand(10, 4)
  output = torch.IntTensor{1, 2, 5, 4, 3, 4, 1, 2, 3, 5}
  
  local feval = function(x)
      self.grad_params:zero()

      -- compute the loss
      local lstm_output, class_predictions, caption_loss = self.image_captioner:forward(input, output)

      -- compute the input gradients with respect to the loss
      local input_grads = self.image_captioner:backward(input, lstm_output, class_predictions, output)

      -- regularization
      caption_loss = caption_loss + 0.5 * self.reg * self.params:norm() ^ 2
      self.grad_params:add(self.reg, self.params)

      return caption_loss, self.grad_params
  end

  diff, DC, DC_est = optim.checkgrad(feval, self.params, 1e-6)
  print("Gradient error for lstm captioner is")
  print(diff)

  assert(diff < self.tol, "Gradient is greater than tolerance")

end

function GradChecks:check_lstm_captioner()
  self.image_captioner = imagelstm.ImageCaptionerLSTM{
    in_dim  = self.in_dim,
    mem_dim = self.mem_dim,
    output_module_fn = self:new_caption_module(),
    criterion = self.criterion,
    num_classes = self.num_classes
  }
  self.params, self.grad_params = self.image_captioner:getParameters()
  input = torch.rand(10, 4)
  output = torch.IntTensor{1, 2, 5, 4, 3, 4, 1, 2, 3, 5}
  
  local feval = function(x)
      self.grad_params:zero()

      -- compute the loss
      local lstm_output, class_predictions, caption_loss = self.image_captioner:forward(input, output)

      -- compute the input gradients with respect to the loss
      local input_grads = self.image_captioner:backward(input, lstm_output, class_predictions, output)

      -- regularization
      caption_loss = caption_loss + 0.5 * self.reg * self.params:norm() ^ 2
      self.grad_params:add(self.reg, self.params)

      return caption_loss, self.grad_params
  end

  diff, DC, DC_est = optim.checkgrad(feval, self.params, 1e-6)
  print("Gradient error for lstm captioner is")
  print(diff)

  assert(diff < self.tol, "Gradient is greater than tolerance")

end

-- adds modules into parallel network from module list
-- requires parallel_net is of type nn.parallel
-- requires module_list is an array of modules that is not null
-- modifies: parallel_net by adding modules into parallel net
function GradChecks:add_modules(parallel_net, module_list)
  assert(parallel_net ~= nil)
  assert(module_list ~= nil)
  for i = 1, #module_list do
    curr_module = module_list[i]
    parallel_net:add(curr_module)
  end
end

function GradChecks:check_lstm_captioner_hidden()
  self.image_captioner = imagelstm.ImageCaptionerLSTM{
    in_dim  = self.in_dim,
    mem_dim = self.mem_dim,
    output_module_fn = self:new_caption_module(),
    criterion = self.criterion,
    num_classes = self.num_classes
  }

  input_emb = torch.IntTensor{1, 2, 3, 4, 5, 2, 3, 1, 4, 5}
  input_feats = torch.rand(104)

  self.proj_layer = imagelstm.EmbedLayer{emb_dim = self.in_dim, 
  num_classes = self.num_classes
  }

  self.hidden_proj_layer = imagelstm.HiddenProjLayer{image_dim = 104, mem_dim = self.mem_dim}

  self.captioner_modules = self.image_captioner:getModules()
  self.combine_modules = self.proj_layer:getModules()
  self.hidden_modules = self.hidden_proj_layer:getModules()

  local modules = nn.Parallel()
  self:add_modules(modules, self.captioner_modules)
  self:add_modules(modules, self.combine_modules)
  self:add_modules(modules, self.hidden_modules)

  self.params, self.grad_params = modules:getParameters()
  output = torch.IntTensor{1, 2, 5, 4, 3, 4, 1, 2, 3, 5}
  
  local feval = function(x)
      self.grad_params:zero()

      local input = self.proj_layer:forward(input_emb)
      local hidden_input = self.hidden_proj_layer:forward(input_feats)
      -- compute the loss
      local lstm_output, class_predictions, caption_loss = 
        self.image_captioner:forward(input, hidden_input, output)

      -- compute the input gradients with respect to the loss
      local input_grads, input_param_grads = 
      self.image_captioner:backward(input, hidden_input, lstm_output, class_predictions, output)

      local input_err = self.proj_layer:backward(input_emb, input_grads)
      local hidden_err = self.hidden_proj_layer:backward(input_feats, input_param_grads)
      -- regularization
      caption_loss = caption_loss + 0.5 * self.reg * self.params:norm() ^ 2
      self.grad_params:add(self.reg, self.params)

      return caption_loss, self.grad_params
  end

  diff, DC, DC_est = optim.checkgrad(feval, self.params, 1e-7)
  print("Gradient error for lstm captioner with hidden input is")
  print(diff)

  assert(diff < self.tol, "Gradient is greater than tolerance")

end


function GradChecks:check_add_module()

  self.add_layer = imagelstm.AddLayer{
    emb_dim = 10,
    image_dim = 10
  }

  self.add_params, self.add_grads = self.add_layer:getParameters()
  local sentence_input = torch.IntTensor{1, 2, 5, 4, 3}
  local image_input = torch.rand(10)
  
  local desired = torch.rand(5, 10)
  self.mse_criterion = nn.MSECriterion()

  local feval = function(x)
      self.add_grads:zero()

      -- compute the loss
      local outputs = self.add_layer:forward(sentence_input, image_input)
      local err = self.mse_criterion:forward(outputs, desired)

      local d_outputs = self.mse_criterion:backward(outputs, desired)
      -- compute the input gradients with respect to the loss
      local input_grads = self.add_layer:backward(sentence_input, image_input, d_outputs)

      return err, self.add_grads
  end

  
  diff, DC, DC_est = optim.checkgrad(feval, self.add_params, 1e-6)
  print("Gradient error for add layer is")
  print(diff)

  assert(diff < self.tol, "Gradient is greater than tolerance")

end


function GradChecks:check_concat_proj_module()
  self.add_layer = imagelstm.ConcatProjLayer{
    emb_dim = 10,
    image_dim = 10,
    proj_dim = 10
  }

  self.add_params, self.add_grads = self.add_layer:getParameters()
  local sentence_input = torch.IntTensor{1, 2, 5, 4, 3}
  local image_input = torch.rand(10)
  
  local desired = torch.rand(5, 20)
  self.mse_criterion = nn.MSECriterion()

  local feval = function(x)
      self.add_grads:zero()

      -- compute the loss
      local outputs = self.add_layer:forward(sentence_input, image_input)
      local err = self.mse_criterion:forward(outputs, desired)

      local d_outputs = self.mse_criterion:backward(outputs, desired)
      -- compute the input gradients with respect to the loss
      local input_grads = self.add_layer:backward(sentence_input, image_input, d_outputs)

      return err, self.add_grads
  end

  
  diff, DC, DC_est = optim.checkgrad(feval, self.add_params, 1e-6)
  print("Gradient error for concat projection module is")
  print(diff)

  assert(diff < self.tol, "Gradient is greater than tolerance")

end

function GradChecks:check_add_module()

  self.add_layer = imagelstm.AddLayer{
    emb_dim = 10,
    image_dim = 10
  }

  self.add_params, self.add_grads = self.add_layer:getParameters()
  local sentence_input = torch.IntTensor{1, 2, 5, 4, 3}
  local image_input = torch.rand(10)
  
  local desired = torch.rand(5, 10)
  self.mse_criterion = nn.MSECriterion()

  local feval = function(x)
      self.add_grads:zero()

      -- compute the loss
      local outputs = self.add_layer:forward(sentence_input, image_input)
      local err = self.mse_criterion:forward(outputs, desired)

      local d_outputs = self.mse_criterion:backward(outputs, desired)
      -- compute the input gradients with respect to the loss
      local input_grads = self.add_layer:backward(sentence_input, image_input, d_outputs)

      return err, self.add_grads
  end

  
  diff, DC, DC_est = optim.checkgrad(feval, self.add_params, 1e-6)
  print("Gradient error for add layer is")
  print(diff)

  assert(diff < self.tol, "Gradient is greater than tolerance")

end


function GradChecks:check_single_add_module()
  self.add_layer = imagelstm.SingleAddLayer{
    emb_dim = 10,
    image_dim = 10,
  }

  self.add_params, self.add_grads = self.add_layer:getParameters()
  local sentence_input = torch.IntTensor{1, 2, 5, 4, 3}
  local image_input = torch.rand(10)
  
  local desired = torch.rand(5, 10)
  self.mse_criterion = nn.MSECriterion()

  local feval = function(x)
      self.add_grads:zero()

      -- compute the loss
      local outputs = self.add_layer:forward(sentence_input, image_input)
      local err = self.mse_criterion:forward(outputs, desired)

      local d_outputs = self.mse_criterion:backward(outputs, desired)
      -- compute the input gradients with respect to the loss
      local input_grads = self.add_layer:backward(sentence_input, image_input, d_outputs)

      return err, self.add_grads
  end

  
  diff, DC, DC_est = optim.checkgrad(feval, self.add_params, 1e-6)
  print("Gradient error for single add module is")
  print(diff)

  assert(diff < self.tol, "Gradient is greater than tolerance")

end

function GradChecks:check_concat_module()

  self.add_layer = imagelstm.ConcatLayer{
    emb_dim = 10,
    image_dim = 10
  }

  self.add_params, self.add_grads = self.add_layer:getParameters()
  local sentence_input = torch.IntTensor{1, 2, 5, 4, 3}
  local image_input = torch.rand(10)
  
  local desired = torch.rand(5, 20)
  self.mse_criterion = nn.MSECriterion()

  local feval = function(x)
      self.add_grads:zero()

      -- compute the loss
      local outputs = self.add_layer:forward(sentence_input, image_input)
      local err = self.mse_criterion:forward(outputs, desired)

      local d_outputs = self.mse_criterion:backward(outputs, desired)
      -- compute the input gradients with respect to the loss
      local input_grads = self.add_layer:backward(sentence_input, image_input, d_outputs)

      return err, self.add_grads
  end

  
  diff, DC, DC_est = optim.checkgrad(feval, self.add_params, 1e-6)
  print("Gradient error for concat module is")
  print(diff)

  assert(diff < self.tol, "Gradient is greater than tolerance")

end

