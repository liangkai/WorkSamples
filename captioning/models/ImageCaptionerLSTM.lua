--[[

  An ImageCaptionerLSTM takes in three things as input: 
  1) an LSTM cell 
  2) an output function for that cell that is the criterion.
  3) an input function that converts input to one that can go into LSTM cell

--]]

local ImageCaptionerLSTM = torch.class('imagelstm.ImageCaptionerLSTM')

function ImageCaptionerLSTM:__init(config)
  -- parameters for lstm cell
  self.gpu_mode = config.gpu_mode
  self.criterion = config.criterion
  self.output_module_fn = config.output_module_fn
  self.lstm_layer =  imagelstm.LSTM_Decoder(config) 

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
function ImageCaptionerLSTM:enable_dropouts()
   enable_sequential_dropouts(self.output_module_fn)
end

-- Disable Dropouts
function ImageCaptionerLSTM:disable_dropouts()
   disable_sequential_dropouts(self.output_module_fn)
end



-- Resets depth to 1
function ImageCaptionerLSTM:reset_depth()
  self.lstm_layer.depth = 0
end


function ImageCaptionerLSTM:zeroGradParameters()
  self.grad_params:zero()
  self.lstm_layer:zeroGradParameters()
end

-- Forward propagate.
-- inputs: T x in_dim tensor, where T is the number of time steps.
-- states: hidden, cell states of LSTM if true, read the input from right to left (useful for bidirectional LSTMs).
-- labels: T x 1 tensor of desired indeces
-- Returns lstm output, class predictions, and error if train, else not error 
function ImageCaptionerLSTM:forward(inputs, hidden_inputs, labels)
    assert(inputs ~= nil)
    assert(hidden_inputs ~= nil)
    assert(labels ~= nil)
    
    local lstm_output = self.lstm_layer:forward(inputs, hidden_inputs, self.reverse)
    local class_predictions = self.output_module_fn:forward(lstm_output)
    local err = self.criterion:forward(class_predictions, labels)

    return lstm_output, class_predictions, err
end

-- Single tick of LSTM Captioner
-- inputs: T x in_dim tensor, where T is the number of time steps.
-- states: hidden, cell states of LSTM
-- labels: T x 1 tensor of desired indeces
-- Returns lstm output, class predictions, and error if train, else not error 
function ImageCaptionerLSTM:tick(inputs, states)
    assert(inputs ~= nil)
    assert(states ~= nil)

    local lstm_output = self.lstm_layer:tick(inputs, states)
    local htable = lstm_output[2]
    local hidden_state
    if self.lstm_layer.num_layers > 1 then 
      hidden_state = htable[self.lstm_layer.num_layers]
    else
      hidden_state = htable
    end
    local class_predictions = self.output_module_fn:forward(hidden_state)
    local lstm_output = self.lstm_layer:tick(inputs, states)
    return lstm_output, class_predictions
end

-- Backpropagate: forward() must have been called previously on the same input.
-- inputs: T x in_dim tensor, where T is the number of time steps.
-- hidden_inputs: {hidden_dim, hidden_tim} tensors
-- reverse: True if reverse input, false otherwise
-- lstm_output: T x num_layers x num_hidden tensor
-- class_predictions: T x 1 tensor of predictions
-- labels: actual labels
-- Returns the gradients with respect to the inputs (in the same order as the inputs).
function ImageCaptionerLSTM:backward(inputs, hidden_inputs, lstm_output, class_predictions, labels)
  assert(inputs ~= nil)
  assert(hidden_inputs ~= nil)
  assert(lstm_output ~= nil)
  assert(class_predictions ~= nil)
  assert(labels ~= nil)

  local output_module_derivs = self.criterion:backward(class_predictions, labels)
  local lstm_output_derivs = self.output_module_fn:backward(lstm_output, output_module_derivs)
  local lstm_input_derivs, hidden_derivs = self.lstm_layer:backward(inputs, hidden_inputs, lstm_output_derivs, self.reverse)

  --print("Backward Differences are", 33 * (end1 - start1), 33 *(end2 - end1), 33 * (end3 - end2))
  return lstm_input_derivs, hidden_derivs
end

-- Sets all networks to gpu mode
function ImageCaptionerLSTM:set_gpu_mode()
  self.criterion:cuda()
  self.output_module_fn:cuda()
  self.lstm_layer:cuda()
end

-- Sets all networks to cpu mode
function ImageCaptionerLSTM:set_cpu_mode()
  self.criterion:double()
  self.output_module_fn:double()
  self.lstm_layer:double()
end

function ImageCaptionerLSTM:getModules() 
  return {self.lstm_layer, self.output_module_fn}
end

function ImageCaptionerLSTM:getParameters()
  return self.params, self.grad_params
end

function ImageCaptionerLSTM:getWeights()
  return self.params
end

