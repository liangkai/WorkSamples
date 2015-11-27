--[[

  An GoogleImageCaptionerLSTM takes in three things as input: 
  1) an LSTM cell 
  2) an output function for that cell that is the criterion.
  3) an input function that converts input to one that can go into LSTM cell

--]]

local GoogleImageCaptionerLSTM = torch.class('imagelstm.GoogleImageCaptionerLSTM')

function GoogleImageCaptionerLSTM:__init(config)
  -- parameters for lstm cell
  self.gpu_mode = config.gpu_mode
  self.criterion        =  config.criterion
  self.output_module_fn = config.output_module_fn
  self.lstm_layer =  imagelstm.LSTM_Decoder(config) 
  self.reverse = false

  local modules = nn.Parallel()
    :add(self.lstm_layer)
    :add(self.output_module_fn)
    
  if self.gpu_mode then
    modules:cuda()
    self.criterion:cuda()
  end
  self.params, self.grad_params = modules:getParameters()
end

-- Enable Dropouts
function GoogleImageCaptionerLSTM:enable_dropouts()
   enable_sequential_dropouts(self.output_module_fn)
end

-- Disable Dropouts
function GoogleImageCaptionerLSTM:disable_dropouts()
   disable_sequential_dropouts(self.output_module_fn)
end



-- Resets depth to 1
function GoogleImageCaptionerLSTM:reset_depth()
  self.lstm_layer.depth = 0
end


function GoogleImageCaptionerLSTM:zeroGradParameters()
  self.grad_params:zero()
  self.lstm_layer:zeroGradParameters()
end

-- Forward propagate.
-- inputs: T + 1 x in_dim tensor, where T is the number of time steps. First time step is for image
-- states: hidden, cell states of LSTM if true, read the input from right to left (useful for bidirectional LSTMs).
-- labels: T x 1 tensor of desired indeces
-- Returns lstm output, class predictions, and error if train, else not error 
function GoogleImageCaptionerLSTM:forward(inputs, hidden_inputs, labels)
    assert(inputs ~= nil)
    assert(hidden_inputs ~= nil)
    assert(labels ~= nil)
    
    local lstm_output = self.lstm_layer:forward(inputs, hidden_inputs, self.reverse)
    local lstm_pred = lstm_output:narrow(1, 2, inputs:size(1) - 1)
    assert(lstm_pred:size(1) == labels:size(1))
    local class_predictions = self.output_module_fn:forward(lstm_pred)
    local err = self.criterion:forward(class_predictions, labels)

    return lstm_output, class_predictions, err
end

-- Single tick of LSTM Captioner
-- inputs: T x in_dim tensor, where T is the number of time steps.
-- states: hidden, cell states of LSTM
-- labels: T x 1 tensor of desired indeces
-- Returns lstm output, class predictions, and error if train, else not error 
function GoogleImageCaptionerLSTM:tick(inputs, states)
    assert(inputs ~= nil)
    assert(states ~= nil)

    local tmp = torch.Tensor(1, inputs:size(1))
    tmp[1] = inputs
    local lstm_output = self.lstm_layer:forward(tmp, states, false)
    local class_predictions = self.output_module_fn:forward(lstm_output)

    --local lstm_output = self.lstm_layer:tick(inputs, states)
    --local ctable, htable = unpack(lstm_output)
    --local hidden_state
    --if self.lstm_layer.num_layers > 1 then 
      --hidden_state = htable[self.lstm_layer.num_layers]
    --else
      --hidden_state = htable
    --end
    --local class_predictions = self.output_module_fn:forward(hidden_state)
    return lstm_output, class_predictions
end

-- Backpropagate: forward() must have been called previously on the same input.
-- inputs: T + 1 x in_dim tensor, where T is the number of time steps.
-- hidden_inputs: {hidden_dim, hidden_tim} tensors
-- reverse: True if reverse input, false otherwise
-- lstm_output: T + 1 x num_layers x num_hidden tensor
-- class_predictions: T x 1 tensor of predictions
-- labels: actual labels
-- Returns the gradients with respect to the inputs (in the same order as the inputs).
function GoogleImageCaptionerLSTM:backward(inputs, hidden_inputs, lstm_output, class_predictions, labels)
  assert(inputs ~= nil)
  assert(hidden_inputs ~= nil)
  assert(lstm_output ~= nil)
  assert(class_predictions ~= nil)
  assert(labels ~= nil)

  local T = class_predictions:size(1)

  -- get predicted derivatives
  local output_module_derivs = self.criterion:backward(class_predictions, labels)
  local lstm_pred = lstm_output:narrow(1, 2, inputs:size(1) - 1)
  local lstm_pred_derivs = self.output_module_fn:backward(lstm_pred, output_module_derivs)

  
  local hidden_dim = lstm_pred_derivs:size(2)

  -- output of lstm derivatives are 1 + predicted derivatives x in_dim (since image fed in once)
  local lstm_output_derivs = self.gpu_mode and torch.CudaTensor(T + 1, hidden_dim):zero()
                              or torch.Tensor(T + 1, hidden_dim):zero()

  lstm_output_derivs:narrow(1, 2, T):copy(lstm_pred_derivs)
  local lstm_input_derivs, hidden_derivs = self.lstm_layer:backward(inputs, hidden_inputs, lstm_output_derivs, self.reverse)

  --print("Backward Differences are", 33 * (end1 - start1), 33 *(end2 - end1), 33 * (end3 - end2))
  return lstm_input_derivs, hidden_derivs
end

-- Sets all networks to gpu mode
function GoogleImageCaptionerLSTM:set_gpu_mode()
  self.criterion:cuda()
  self.output_module_fn:cuda()
  self.lstm_layer:cuda()
end

-- Sets all networks to cpu mode
function GoogleImageCaptionerLSTM:set_cpu_mode()
  self.criterion:double()
  self.output_module_fn:double()
  self.lstm_layer:double()
end

function GoogleImageCaptionerLSTM:getModules() 
  return {self.lstm_layer, self.output_module_fn}
end

function GoogleImageCaptionerLSTM:getParameters()
  return self.params, self.grad_params
end

function GoogleImageCaptionerLSTM:getWeights()
  return self.params
end

