--[[
 Long Short-Term Memory that returns hidden states after all input fed in,
 You're responsible for putting initializing hidden/cell states
--]]

local LSTM, parent = torch.class('imagelstm.LSTM_Encoder', 'nn.Module')

function LSTM:__init(config)
  parent.__init(self)
  assert(config.in_dim ~= nil, "Input dim to lstm must be specified")
  assert(config.mem_dim ~= nil, "Memory dim to lstm must be specified")
  assert(config.num_layers ~= nil, "Number of layers to lstm must be specified")
  assert(config.gpu_mode ~= nil, "Gpu mode of lstm must be specified")
  
  self.in_dim = config.in_dim
  self.mem_dim = config.mem_dim
  self.num_layers = config.num_layers
  self.gpu_mode = config.gpu_mode

  self.master_cell = self:new_cell()
  self.depth = 0
  self.cells = {}  -- table of cells in a roll-out
  self.tensors = {}  -- table of tensors for faster lookup
  self.back_tensors = {} -- table of tensors for backprop

  -- initial (t = 0) states for forward propagation and initial error signals
  -- for backpropagation
  local ctable_init, ctable_grad, htable_init, htable_grad
  if self.num_layers == 1 then
    ctable_init = torch.zeros(self.mem_dim)
    htable_init = torch.zeros(self.mem_dim)
    ctable_grad = torch.zeros(self.mem_dim)
    htable_grad = torch.zeros(self.mem_dim)

    if self.gpu_mode then
      ctable_init:cuda()
      htable_init:cuda()
      ctable_grad:cuda()
      htable_grad:cuda()
    end
  else
    ctable_init, ctable_grad, htable_init, htable_grad = {}, {}, {}, {}
    for i = 1, self.num_layers do
      ctable_init[i] = torch.zeros(self.mem_dim)
      htable_init[i] = torch.zeros(self.mem_dim)
      ctable_grad[i] = torch.zeros(self.mem_dim)
      htable_grad[i] = torch.zeros(self.mem_dim)
    end
    if self.gpu_mode then 
      for i = 1, self.num_layers do
        ctable_init[i]:cuda()
        htable_init[i]:cuda()
        ctable_grad[i]:cuda()
        htable_grad[i]:cuda()
      end
    end
  end

  -- precreate outputs for faster performance
  for i = 1, 100 do
    if self.gpu_mode then 
      self.tensors[i] = torch.FloatTensor(i, self.mem_dim):zero():cuda()
      self.back_tensors[i] = torch.FloatTensor(i, self.in_dim):zero():cuda()
    else
      self.tensors[i] = torch.DoubleTensor(i, self.mem_dim):zero()
      self.back_tensors[i] = torch.DoubleTensor(i, self.in_dim):zero()
    end
  end

  if self.gpu_mode then
    self.initial_values = {ctable_init, htable_init}
    self.gradInput = {
      torch.zeros(self.in_dim):cuda(),
      ctable_grad,
      htable_grad
    }
  else 
    self.initial_values = {ctable_init, htable_init}
    self.gradInput = {
      torch.zeros(self.in_dim),
      ctable_grad,
      htable_grad
    }
  end

end

-- Instantiate a new LSTM cell.
-- Each cell shares the same parameters, but the activations of their constituent
-- layers differ.
function LSTM:new_cell()
 return self:fast_lstm(self.in_dim, self.mem_dim)
 --return self:old_lstm()
end

function LSTM:old_lstm()
  local input = nn.Identity()()
  local ctable_p = nn.Identity()()
  local htable_p = nn.Identity()()

  -- multilayer LSTM
  local htable, ctable = {}, {}
  for layer = 1, self.num_layers do
    local h_p = (self.num_layers == 1) and htable_p or nn.SelectTable(layer)(htable_p)
    local c_p = (self.num_layers == 1) and ctable_p or nn.SelectTable(layer)(ctable_p)

    local new_gate = function()
      local in_module = (layer == 1)
        and nn.Linear(self.in_dim, self.mem_dim)(input)
        or  nn.Linear(self.mem_dim, self.mem_dim)(htable[layer - 1])
      return nn.CAddTable(){
        in_module,
        nn.Linear(self.mem_dim, self.mem_dim)(h_p)
      }
    end

    -- input, forget, and output gates
    local i = nn.Sigmoid()(new_gate())
    local f = nn.Sigmoid()(new_gate())
    local update = nn.Tanh()(new_gate())

    -- update the state of the LSTM cell
    ctable[layer] = nn.CAddTable(){
      nn.CMulTable(){f, c_p},
      nn.CMulTable(){i, update}
    }

    if self.gate_output then
      local o = nn.Sigmoid()(new_gate())
      htable[layer] = nn.CMulTable(){o, nn.Tanh()(ctable[layer])}
    else
      htable[layer] = nn.Tanh()(ctable[layer])
    end
  end

  -- if LSTM is single-layered, this makes htable/ctable Tensors (instead of tables).
  -- this avoids some quirks with nngraph involving tables of size 1.
  htable, ctable = nn.Identity()(htable), nn.Identity()(ctable)
  local cell = nn.gModule({input, ctable_p, htable_p}, {ctable, htable})

  if self.gpu_mode then
    cell:cuda()
  end

  -- share parameters
  if self.master_cell then
    share_params(cell, self.master_cell, 'weight', 'bias', 'gradWeight', 'gradBias')
  end
  return cell
end

--[[ 
Efficient LSTM in Torch using nngraph library. This code was optimized 
by Justin Johnson (@jcjohnson) based on the trick of batching up the 
LSTM GEMMs, as also seen in my efficient Python LSTM gist.
--]]
 
 function LSTM:fast_lstm(input_size, rnn_size)
  local input = nn.Identity()()
  local ctable_p = nn.Identity()()
  local htable_p = nn.Identity()()

  -- multilayer LSTM
  local htable, ctable = {}, {}
  for layer = 1, self.num_layers do
    local h_p = (self.num_layers == 1) and htable_p or nn.SelectTable(layer)(htable_p)
    local c_p = (self.num_layers == 1) and ctable_p or nn.SelectTable(layer)(ctable_p)

    local new_gate = function()
      local in_module = (layer == 1)
        and nn.Linear(input_size, 4 * rnn_size)(input)
        or  nn.Linear(rnn_size, 4 * rnn_size)(htable[layer - 1])
      return nn.CAddTable(){
        in_module,
        nn.Linear(rnn_size, 4 * rnn_size)(h_p)
      }
    end

    local all_input_sums = new_gate()
    local sigmoid_chunk = nn.Narrow(1, 1, 3 * rnn_size)(all_input_sums)
    sigmoid_chunk = nn.Sigmoid()(sigmoid_chunk)
    local in_gate = nn.Narrow(1, 1, rnn_size)(sigmoid_chunk)
    local forget_gate = nn.Narrow(1, rnn_size + 1, rnn_size)(sigmoid_chunk)
    local out_gate = nn.Narrow(1, 2 * rnn_size + 1, rnn_size)(sigmoid_chunk)
 
    local in_transform = nn.Narrow(1, 3 * rnn_size + 1, rnn_size)(all_input_sums)
    in_transform = nn.Tanh()(in_transform)
 
    ctable[layer] = nn.CAddTable()({
      nn.CMulTable()({forget_gate, c_p}),
      nn.CMulTable()({in_gate,     in_transform})
    })
    htable[layer] = nn.CMulTable()({out_gate, nn.Tanh()(ctable[layer])})
  end

  -- if LSTM is single-layered, this makes htable/ctable Tensors (instead of tables).
  -- this avoids some quirks with nngraph involving tables of size 1.
  htable, ctable = nn.Identity()(htable), nn.Identity()(ctable)
  local cell = nn.gModule({input, ctable_p, htable_p}, {ctable, htable})

  if self.gpu_mode then
    cell:cuda()
  end
  
  -- share parameters
  if self.master_cell then
    share_params(cell, self.master_cell, 'weight', 'bias', 'gradWeight', 'gradBias')
  end
  return cell
end

-- Forward propagate.
-- inputs: T x in_dim tensor, where T is the number of time steps.
-- reverse: if true, read the input from right to left (useful for bidirectional LSTMs).
-- Returns T x mem_dim tensor, all the intermediate hidden states of the LSTM. If multilayered,
-- returns output only of last memory layer
function LSTM:forward(inputs, hidden_inputs, reverse)
  local size = inputs:size(1)
  for t = 1, size do
    local input = reverse and inputs[size - t + 1] or inputs[t]
    self.depth = self.depth + 1
    local cell = self.cells[self.depth]

    if cell == nil then
      --print("Cells are null at depth ", self.depth)
      cell = self:new_cell()
      self.cells[self.depth] = cell
    end

    local prev_output
    if self.depth > 1 then
      prev_output = self.cells[self.depth - 1].output
    else
      prev_output = hidden_inputs
    end

    local cell_inputs = {input, prev_output[1], prev_output[2]}
    local outputs = cell:forward(cell_inputs)
    local htable = outputs[2]
    
    if self.num_layers == 1 then
      self.output = htable
    else
      self.output = htable[self.num_layers]
    end
  end
  return self.output
end


-- Backpropagate. forward() must have been called previously on the same input.
-- inputs: T x in_dim tensor, where T is the number of time steps.
-- grad_outputs: T x num_layers x mem_dim tensor.
-- reverse: if true, read the input from right to left.
-- Returns the gradients with respect to the inputs (in the same order as the inputs).
function LSTM:backward(inputs, hidden_inputs, grad_outputs, reverse)
  assert(inputs ~= nil, "inputs are nil")
  assert(hidden_inputs ~= nil, "hidden inputs are null")
  assert(grad_outputs ~= nil, "grad outputs are null")
  assert(reverse ~= nil, "reverse is null")

  local size = inputs:size(1)
  if self.depth == 0 then
    error("No cells to backpropagate through")
  end

  
  local input_grads = self.back_tensors[size]
 
  if input_grads == nil then
    if self.gpu_mode then
      self.back_tensors[size] = torch.FloatTensor(inputs:size()):cuda()
    else
      self.back_tensors[size] = torch.DoubleTensor(inputs:size())
    end
    input_grads = self.back_tensors[size]
  end

  for t = size, 1, -1 do
    local input = reverse and inputs[size - t + 1] or inputs[t]
    local grad_output = reverse and grad_outputs[size - t + 1] or grad_outputs[t]
    local cell = self.cells[self.depth]
    local grads = {self.gradInput[2], self.gradInput[3]}
    if self.num_layers == 1 then
      grads[2]:add(grad_output)
    else
      grads[2][self.num_layers]:add(grad_output)
    end

    local prev_output = (self.depth > 1) and self.cells[self.depth - 1].output
                                         or hidden_inputs
    self.gradInput = cell:backward({input, prev_output[1], prev_output[2]}, grads)
    if reverse then
      input_grads[size - t + 1] = self.gradInput[1]
    else
      input_grads[t] = self.gradInput[1]
    end
    if self.depth == 1 then
      if self.num_layers == 1 then
        self.initial_values[1]:copy(self.gradInput[2])
        self.initial_values[2]:copy(self.gradInput[3])
      else 
        for i = 1, self.num_layers do 
          self.initial_values[1][i]:copy(self.gradInput[2][i])
          self.initial_values[2][i]:copy(self.gradInput[3][i])
        end
      end
    end
    self.depth = self.depth - 1
  end
  self:forget() -- important to clear out state
  return input_grads, self.initial_values
end

function LSTM:share(lstm, ...)
  if self.in_dim ~= lstm.in_dim then error("LSTM input dimension mismatch") end
  if self.mem_dim ~= lstm.mem_dim then error("LSTM memory dimension mismatch") end
  if self.num_layers ~= lstm.num_layers then error("LSTM layer count mismatch") end
  if self.gate_output ~= lstm.gate_output then error("LSTM output gating mismatch") end
  share_params(self.master_cell, lstm.master_cell, ...)
end

function LSTM:zeroGradParameters()
  self.master_cell:zeroGradParameters()
end

function LSTM:getModules()
  return {self.master_cell}
end

function LSTM:parameters()
  return self.master_cell:parameters()
end

-- Clear saved gradients
function LSTM:forget()
  self.depth = 0
  for i = 1, #self.gradInput do
    local gradInput = self.gradInput[i]
    if type(gradInput) == 'table' then
      for _, t in pairs(gradInput) do t:zero() end
    else
      self.gradInput[i]:zero()
    end
  end
end


