--[[

  Image Captioning using an LSTM and a softmax for the language model

--]]

local GpuChecks = torch.class('imagelstm.GpuChecks')

function GpuChecks:__init(config)
  require('cutorch')
  require('cunn')

  print("Checking gpu stuff with in_dim ", config.in_dim, " mem_dim ", config.mem_dim, 
    " num_classes ", config.num_classes)
  self.in_dim = config.in_dim
  self.mem_dim = config.mem_dim
  self.num_classes = config.num_classes

  self.image_captioner = imagelstm.ImageCaptionerLSTM{
    gpu_mode = false,
    in_dim  = self.in_dim,
    mem_dim = self.mem_dim,
    output_module_fn = self:new_caption_module(),
    criterion = nn.ClassNLLCriterion(),
  }

  self.gpu_image_captioner = imagelstm.ImageCaptionerLSTM{
    gpu_mode = true,
    in_dim  = self.in_dim,
    mem_dim = self.mem_dim,
    output_module_fn = self:new_caption_module(),
    criterion = nn.ClassNLLCriterion(),
  }

end

function GpuChecks:new_caption_module()
  local caption_module = nn.Sequential()

  caption_module
    :add(nn.Linear(self.mem_dim, self.num_classes))
    :add(nn.LogSoftMax())
  return caption_module
end

function GpuChecks:check_gpu()
  self:check_lstm_captioner()
  self:check_lstm_full_layer()
  self:check_torch_tensor()
  self:check_lstm_create_cell()
  self:check_lstm_cell()
end

function GpuChecks:check_torch_tensor()
  local num_iter = 200
  local start_time = sys.clock()
  for i = 1, num_iter do
      local tensor = torch.Tensor(self.num_classes, 100)
  end
  local end_time = sys.clock()

  print("Cpu time for creating torch tensor")
  print((end_time - start_time) / num_iter)

  local start_time = sys.clock()
  for i = 1, num_iter do
      local tensor = torch.Tensor(self.num_classes, 100):cuda()
  end
  local end_time = sys.clock()

  print("Gpu time for creating torch tensor")
  print((end_time - start_time) / num_iter) 
end

function GpuChecks:check_lstm_create_cell()
  local input = torch.rand(self.in_dim)
  local num_iter = 100

  local lstm_gpu_layer = imagelstm.LSTM_Full{
    gpu_mode = true,
    in_dim  = self.in_dim,
    mem_dim = self.mem_dim,
  }

  local lstm_cpu_layer = imagelstm.LSTM_Full{
    gpu_mode = false,
    in_dim  = self.in_dim,
    mem_dim = self.mem_dim,
  }
  
  gpu_cell = lstm_gpu_layer:new_cell()

  local lstm_cpu_input = {input, lstm_cpu_layer.initial_values[1], 
                        lstm_cpu_layer.initial_values[2]}

  local lstm_gpu_input = {input:cuda(), lstm_gpu_layer.initial_values[1], 
                        lstm_gpu_layer.initial_values[2]}

  local start_time = sys.clock()
  for i = 1, num_iter do
      local cpu_cell = lstm_cpu_layer:new_cell()
  end
  local end_time = sys.clock()

  print("Cpu time for creating lstm cell")
  print((end_time - start_time) / num_iter)

  local start_time = sys.clock()
  for i = 1, num_iter do
      local gpu_cell = lstm_gpu_layer:new_cell()
  end
  local end_time = sys.clock()

  print("Gpu time for creating lstm cell")
  print((end_time - start_time) / num_iter)

end

function GpuChecks:check_lstm_cell()
  local input = torch.rand(self.in_dim)
  local num_iter = 300

  local lstm_gpu_layer = imagelstm.LSTM_Full{
    gpu_mode = true,
    in_dim  = self.in_dim,
    mem_dim = self.mem_dim,
  }

  local lstm_cpu_layer = imagelstm.LSTM_Full{
    gpu_mode = false,
    in_dim  = self.in_dim,
    mem_dim = self.mem_dim,
  }

  cpu_cell = lstm_cpu_layer:new_cell()
  gpu_cell = lstm_gpu_layer:new_cell()

  local start_time = sys.clock()

  for i = 1, num_iter do
      cpu_cell:forward(lstm_cpu_input)
  end
  local end_time = sys.clock()

  print("Cpu time for forwarding lstm cell")
  print((end_time - start_time) / num_iter)

  local lstm_gpu_input = {input:cuda(), lstm_gpu_layer.initial_values[1], 
                            lstm_gpu_layer.initial_values[2]}
  local start_time = sys.clock()
  for i = 1, num_iter do
      gpu_cell:forward(lstm_gpu_input)
  end
  local end_time = sys.clock()

  print("Gpu time for forwarding lstm cell")
  print((end_time - start_time) / num_iter)

end

function GpuChecks:check_lstm_full_layer()
  local input = torch.rand(10, self.in_dim)
  local num_iter = 20

  local lstm_gpu_layer = imagelstm.LSTM_Full{
    gpu_mode = true,
    in_dim  = self.in_dim,
    mem_dim = self.mem_dim,
  }

  local lstm_cpu_layer = imagelstm.LSTM_Full{
    gpu_mode = false,
    in_dim  = self.in_dim,
    mem_dim = self.mem_dim,
  }

  local cpu_time = self:check_cpu_speed(input, nil, lstm_cpu_layer, num_iter)

  print("Cpu time for LSTM cell network is")
  print(cpu_time)

  local gpu_time = self:check_gpu_speed(input, nil, lstm_gpu_layer, num_iter)

  print ("Gpu time for LSTM cell network is")
  print(gpu_time)
end

function GpuChecks:check_lstm_captioner()
  local input = torch.rand(10, self.in_dim)
  local output = torch.IntTensor(10)
  for i = 1, 10 do
    output[i] = i
  end

  local num_iter = 20

  local cpu_time = self:check_captioner_cpu_speed(input, output, self.image_captioner, num_iter)
  print("Cpu time for image captioner is")
  print(cpu_time)

  local gpu_time = self:check_captioner_gpu_speed(input, output:cuda(), self.gpu_image_captioner, num_iter)

  print ("Gpu time for image captioner is")
  print(gpu_time)
end

function GpuChecks:check_nn_module()
  local inputs = torch.rand(1000)
  local net = nn.Linear(1000, 5000)
  local num_iter = 1000

  local cpu_time = self:check_cpu_speed(inputs, nil, net, num_iter)
  local gpu_time = self:check_gpu_speed(inputs, nil, net, num_iter)

  print("Cpu time for linear model is ")
  print(cpu_time)

  print ("Gpu time for linear model is ")
  print(gpu_time)
end

-- Checks how fast CPU speed is for captioner
function GpuChecks:check_captioner_cpu_speed(inputs, labels, nnet, num_iter)
  local start_time = sys.clock()
  for i = 1, num_iter do
      lstm_output, class_predictions = nnet:forward(inputs, labels)
      res = nnet:backward(inputs, lstm_output, class_predictions, labels)
  end
  local end_time = sys.clock()
  return (end_time - start_time) / num_iter
end

-- Checks how fast GPU speed is for neural net
function GpuChecks:check_captioner_gpu_speed(inputs, labels, nnet, num_iter)
  local inputs = inputs:cuda()

  if labels ~= nil then
    labels = labels
  end
  local start_time = sys.clock()
  for i = 1, num_iter do
      lstm_output, class_predictions = nnet:forward(inputs, labels)
      res = nnet:backward(inputs, lstm_output, class_predictions, labels)
  end
  local end_time = sys.clock()
  return (end_time - start_time) / num_iter
end

-- Checks how fast CPU speed is for neural net
function GpuChecks:check_cpu_speed(inputs, labels, nnet, num_iter)
  local start_time = sys.clock()
  for i = 1, num_iter do
      res = nnet:forward(inputs, labels)
      tmp = nnet:backward(inputs, res)
  end
  local end_time = sys.clock()
  return (end_time - start_time) / num_iter
end

-- Checks how fast GPU speed is for neural net
function GpuChecks:check_gpu_speed(inputs, labels, nnet, num_iter)
  local inputs = inputs:cuda()

  if labels ~= nil then
    labels = labels:cuda()
  end
  local start_time = sys.clock()
  for i = 1, num_iter do
      tmp = nnet:forward(inputs, labels)
      res = nnet:backward(inputs, tmp)
  end
  local end_time = sys.clock()
  return (end_time - start_time) / num_iter
end



