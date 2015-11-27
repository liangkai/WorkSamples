--[[

  Image Captioning using an LSTM and a softmax for the language model

--]]

local CpuChecks = torch.class('imagelstm.CpuChecks')

function CpuChecks:__init(config)
  local in_dim = 400
  local mem_dim = 1500
  local num_classes = 4000
  local criterion = nn.ClassNLLCriterion()

  self.image_captioner = imagelstm.ImageCaptionerLSTM{
    in_dim  = in_dim,
    mem_dim = mem_dim,
    output_module_fn = self:new_caption_module(),
    criterion = criterion,
  }

end

function CpuChecks:new_caption_module()
  local caption_module = nn.Sequential()
  caption_module
    :add(nn.Linear(1500, 4000))
    :add(nn.LogSoftMax())
  return caption_module
end
function CpuChecks:check_cpu()
	self:check_lstm_full_layer()
end

function CpuChecks:check_lstm_full_layer()
  local in_dim = 300
  local mem_dim = 1500
  local input = torch.rand(10, 300)
  local num_iter = 20

  local lstm_cpu_layer = imagelstm.LSTM_Full{
    gpu_mode = false,
    in_dim  = in_dim,
    mem_dim = mem_dim,
  }

  local cpu_time = self:check_cpu_speed(input, nil, lstm_cpu_layer, num_iter)

  print("Cpu time for lstm full layer is is")
  print(cpu_time)
end

function CpuChecks:check_lstm_captioner()
  local input = torch.rand(10, 400)
  local output = torch.IntTensor(10)
  for i = 1, 10 do
    output[i] = i
  end

  local num_iter = 20

  local cpu_time = self:check_cpu_speed(input, output, self.image_captioner, num_iter)
  print("Cpu time for image captioner is")
  print(cpu_time)
end

function CpuChecks:check_nn_module()
  local inputs = torch.rand(1000)
  local net = nn.Linear(1000, 5000)
  local num_iter = 1000

  local cpu_time = self:check_cpu_speed(inputs, nil, net, num_iter)
end

-- Checks how fast CPU speed is for neural net
function CpuChecks:check_cpu_speed(inputs, labels, nnet, num_iter)
  local start_time = sys.clock()
  for i = 1, num_iter do
      res = nnet:forward(inputs, labels)
      back = nnet:backward(inputs, res)
  end
  local end_time = sys.clock()
  return (end_time - start_time) / num_iter
end





