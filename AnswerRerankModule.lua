--[[

  AnswerRerankModule learns to rerank candidate answers based on the cosine similarity
  of the question and answer

--]]

local AnswerRerankModule = torch.class('dmn.AnswerRerankModule')

function AnswerRerankModule:__init(config)
  assert(config.gpu_mode ~= nil, "Must specify gpu mode for config")
  self.config = config;
  self.gpu_mode = config.gpu_mode

  -- to scale cosine similarity
  self.eps = 1e-4

  -- to scale cosine
  self.lambda = 5;

  self:init_layers()
  dmn.logger:print("Created a new AnswerRerank layer")
end

-- Initializes all the layers of the sequence to sequence model
function AnswerRerankModule:init_layers()

  -- computes cosine similarity, multiplies by lambda, and then takes softmax
  local question = nn.Identity()()
  -- candidates are table, so need to transform them into tensor
  local candidates = nn.Identity()()

  -- TODO: need special jointable that concatenates jagged matrices
  local tensor_candidates = dmn.JoinTable(2)(candidates)
  local similarities = dmn.SmoothCosineSimilarity(self.eps)({tensor_candidates, question})
  local softmax_prob = nn.LogSoftMax()(nn.MulConstant(self.lambda)(similarities))
  
  self.network = nn.gModule({question, candidates}, {softmax_prob})

  self.criterion = nn.ClassNLLCriterion()

  if self.gpu_mode then 
    print("Moving network and criterion for AnswerRerankModule to gpu mode")
    self.network:cuda()
    self.criterion:cuda()
  end
end

-- Returns an array of the modules for this layer
function AnswerRerankModule:getModules() 
  return {}
end

function AnswerRerankModule:forget()
end

-- Set all of the network parameters to gpu mode
function AnswerRerankModule:set_gpu_mode()
  self.gpu_mode = true
  self.network:cuda()
  self.criterion:cuda()
end

function AnswerRerankModule:set_cpu_mode()
  self.gpu_mode = false
  self.network:double()
  self.criterion:double()
end

-- enables dropouts on all layers
function AnswerRerankModule:enable_dropouts()
end

-- disables dropouts on all layers
function AnswerRerankModule:disable_dropouts()
end

-- Forward propagate. Computes cosine similarity between question (input 1), 
-- and all input2s.
-- input1: n dimensional tensor
-- input2: Txn dimensional tensor
function AnswerRerankModule:forward(question_lsr, candidate_answers_lsr, desired_index)
  assert(question_lsr ~= nil, "Must pass input1 to rerank module")
  assert(candidate_answers_lsr ~= nil, "Must pass input2 to rerank module")
  assert(desired_index ~= nil, "Must pass desired index to rerank module")

  self.probabilities = self.network:forward({question_lsr, candidate_answers_lsr})
  local loss = self.criterion:forward(self.probabilities, desired_index)
  return loss
end

-- Backpropagate.
-- input1: num_layers x mem_dim tensor which represents original memory of AnswerRerankModule
-- input2: T size IntTensor, where T is the number of time steps. Represents word indeces
-- desired_indices: T size Tensor, where T is number of time steps. Represents indices we want to predict
-- returns error with respect to the memory state of the answer module.
function AnswerRerankModule:backward(question_lsr, candidate_answers_lsr, desired_index)
  assert(question_lsr ~= nil, "Must specify first inputs to backprop")
  assert(candidate_answers_lsr ~= nil, "Must specify input2 to backprop")
  assert(desired_index ~= nil, "Must specify desired index")

  self.prob_errs = self.criterion:backward(self.probabilities, desired_index)
  self.cosine_sim_errs = self.network:backward({question_lsr, candidate_answers_lsr}, self.prob_errs)
  self.input_errs = self.cosine_sim_errs[2]
  self.question_err = self.cosine_sim_errs[1]

  return {self.question_err, self.input_errs}
 end

function AnswerRerankModule:grad_check()
end

-- Predicts the beam_size most likely answers for the given question and candidate answers
-- questions_lsr: latent semantic representation of question
-- candidate_answers_lsr: list of latent semantic representations of candidate answers
-- beam_size: number of answers to return
-- Returns best indices of predictions
function AnswerRerankModule:predict(question_lsr, candidate_answers_lsr, beam_size)
  assert(question_lsr ~= nil, "Question lsr must be specified")
  assert(candidate_answers_lsr ~= nil, "Candidate answers lsr must be specified")
  assert(beam_size ~= nil, "Must specify beam size for items")
  assert(beam_size > 0, "Beam size must be positive")
  -- join them together and compute the probabilities
  self.probabilities = self.network:forward({question_lsr, candidate_answers_lsr})

  local best_indices = topkargmax(self.probabilities, beam_size)

  local indices_and_prob = {}
  for i = 1, #best_indices do
    local best_index = best_indices[i]
    indices_and_prob[i] = {self.probabilities[best_index], best_index}
  end

  return indices_and_prob

  ---print("BEST INDEX FOR PREDICTION", best_indices[1])
  --print("SIMILARITIES", self.similarities[best_indices[1]])
  --print("PROBABILITIES", self.probabilities[best_indices[1]])
  --print(self.probabilities[best_indices[2]])

end

function AnswerRerankModule:print_config()
  printf('%-25s = %d\n', 'num params for answer rerank module', 0)
  printf('using cosine distance as similarity module')
end

