--[[

  DSSM_Network learns to rank candidate answers based on DSSM similarity 
  between candidate answers and a question
  
--]]

local DSSM_Network = torch.class('dmn.DSSM_Network')

function DSSM_Network:__init(config) 
  config.char_level = true
  config.pad_size = 3
  
  self:check_config(config)

    -- make sure there is no rep exposure
  self.config = dmn.functions.deepcopy(config)
  self.pad_size = config.pad_size
  self.char_level = config.char_level
  self.input_vocab = config.i_vocab
  self.question_vocab = config.q_vocab
  self.gpu_mode = config.gpu_mode
  self.model_epoch = config.model_epoch or 0;

  self:init_layers(config)

  self.batch_size = config.batch_size or 100
  self.optim_method_string = config.optim_method_string or "rmsprop"
  if self.optim_method_string == 'adagrad' then
    self.optim_method = optim.adagrad
  elseif self.optim_method_string == 'rmsprop' then
    self.optim_method = optim.rmsprop
  else
    self.optim_method = optim.sgd
  end

  self.optim_state = config.optim_state
  --self.reg = 1e-4;

  self.reverse = false

  local modules = nn.Parallel()
  for i = 1, #self.layers do
    dmn.logger:print(self.layers[i])
    local curr_modules = self.layers[i]:getModules()
    add_modules(modules, curr_modules)
  end

  if self.gpu_mode then 
    self:set_gpu_mode()
  end

  dmn.logger:print("==== Modules we're optimizing for entire DSSM network ====")
  dmn.logger:print(modules)
  self.params, self.grad_params = modules:getParameters()

end

-- Makes sure that all parameters are specified for this dmn network
function DSSM_Network:check_config(config)
  assert(config.dssm_type ~= nil, "Must specify dssm type to use")
  assert(config.char_level ~= nil, "Must specify whether to use char level or not")
  assert(config.pad_size ~= nil, "Must specify whether to use pad size or not")

  assert(config.gpu_mode ~= nil, "Must specify gpu mode")
  assert(config.i_vocab ~= nil, "Must specify input vocab")
  assert(config.q_vocab ~= nil, "Must specify question vocab")
  assert(config.optim_state ~= nil, "Must specify optim state")
  -- question parameters
  assert(config.question_emb_dim ~= nil, "Must specify question embed dimensions")
  assert(config.question_hidden_dim ~= nil, "Must specify question hidden dimensions")
  assert(config.question_out_dim ~= nil, "Must specify output dimension for question embeddings (of dssm)")
  assert(config.question_in_stride ~= nil, "Must specify question input stride")
  assert(config.question_in_kernel_width ~= nil, "Must specify question input kernel width")
  assert(config.question_hidden_kernel_width ~= nil, "Must specify question hidden kernel width")
  assert(config.question_hidden_stride ~= nil, "Must specify question hidden stride")
  assert(config.question_out_kernel_width ~= nil, "Must specify question output kernel width")
  assert(config.question_out_stride ~= nil, "Must specify question hidden stride")
  assert(config.question_num_classes ~= nil, "Must specify number of classes")
  assert(config.question_dropout_prob ~= nil, "Must specify dropout probability")
  assert(config.question_dropout ~= nil, "Must specify whether to use dropout or not")

  -- input parameters
  assert(config.input_emb_dim ~= nil, "Must specify question embed dimensions")
  assert(config.input_hidden_dim ~= nil, "Must specify input hidden dimensions")
  assert(config.input_out_dim ~= nil, "Must specify output dimension for question embeddings (of dssm)")
  assert(config.input_in_stride ~= nil, "Must specify question input stride")
  assert(config.input_in_kernel_width ~= nil, "Must specify question input kernel width")
  assert(config.input_hidden_kernel_width ~= nil, "Must specify question hidden kernel width")
  assert(config.input_hidden_stride ~= nil, "Must specify question hidden stride")
  assert(config.input_out_kernel_width ~= nil, "Must specify question hidden kernel width")
  assert(config.input_out_stride ~= nil, "Must specify question hidden stride")
  assert(config.input_num_classes ~= nil, "Must specify number of classes")
  assert(config.input_dropout_prob ~= nil, "Must specify dropout probability")
  assert(config.input_dropout ~= nil, "Must specify whether to use dropout or not")
end

-- enables dropouts on all layers
function DSSM_Network:enable_dropouts()
  dmn.logger:print("====== Enabling dropouts ======")
  dmn.functions.enable_dropouts(self.layers)
  dmn.functions.enable_dropouts(self.input_dssm_layers)
  dmn.functions.enable_dropouts(self.input_embed_layers)
end

-- disables dropouts on all layers
function DSSM_Network:disable_dropouts()
  dmn.logger:print("====== Disabling dropouts ======")
  dmn.functions.disable_dropouts(self.layers)
  dmn.functions.disable_dropouts(self.input_dssm_layers)
  dmn.functions.disable_dropouts(self.input_embed_layers)
end

-- enables gpus on dmn network
function DSSM_Network:set_gpu_mode()
  dmn.logger:print("====== Setting gpu mode ======")
  dmn.functions.set_gpu_mode(self.layers)
  dmn.functions.set_gpu_mode(self.input_dssm_layers)
  dmn.functions.set_gpu_mode(self.input_embed_layers)
end

-- converts to cpus on dmn network
function DSSM_Network:set_cpu_mode()
  dmn.logger:print("====== Setting cpu mode ======")
  dmn.functions.set_cpu_mode(self.layers)
  dmn.functions.set_cpu_mode(self.input_dssm_layers)
  dmn.functions.set_cpu_mode(self.input_embed_layers)
end

-- Initializes all the layers of the dynamic memory network
function DSSM_Network:init_layers(config)
  self.answer_layer = self:new_answer_layer(config)
  self.question_embed_layer = self:new_question_embed_layer(config)
  self.question_dssm_layer = self:new_question_dssm_layer(config)

  self.master_input_embed_layer = self:new_input_embed_layer(config)
  self.master_input_dssm_layer = self:new_input_dssm_layer(config)

  self.input_dssm_layers = {}
  self.input_embed_layers = {}

  self.layers = {self.answer_layer, 
                self.question_embed_layer,
                self.master_input_embed_layer,
                self.question_dssm_layer,
                self.master_input_dssm_layer}
end

-- Returns new question dssm layer corresponding to config
function DSSM_Network:new_answer_layer(config)
  local answer_layer = dmn.AnswerRerankModule{
                        gpu_mode = config.gpu_mode
                        }
  return answer_layer
end

-- Returns new question dssm layer corresponding to config
function DSSM_Network:new_question_dssm_layer(config)
  local question_layer = dmn.DSSM_Layer{
              dssm_type = config.dssm_type,
              gpu_mode = config.gpu_mode,
              in_dim = config.question_emb_dim,
              hidden_dim = config.question_hidden_dim,
              out_dim = config.question_out_dim,
              in_stride = config.question_in_stride,
              in_kernel_width = config.question_in_kernel_width,
              hidden_stride = config.question_hidden_stride,
              hidden_kernel_width = config.question_hidden_kernel_width,
              out_stride = config.question_out_stride,
              out_kernel_width = config.question_out_kernel_width
            }
  return question_layer
end

-- Returns new input dssm layer corresponding to config
function DSSM_Network:new_input_dssm_layer(config)
  assert(config ~= nil, "Must specify config for input dssm layer")
  local input_layer = dmn.DSSM_Layer{
              dssm_type = config.dssm_type,
              gpu_mode = config.gpu_mode,
              in_dim = config.input_emb_dim,
              hidden_dim = config.input_hidden_dim,
              out_dim = config.input_out_dim,
              in_stride = config.input_in_stride,
              in_kernel_width = config.input_in_kernel_width,
              hidden_stride = config.input_hidden_stride,
              hidden_kernel_width = config.input_hidden_kernel_width,
              out_stride = config.input_out_stride,
              out_kernel_width = config.input_out_kernel_width
            }
  if self.master_input_dssm_layer ~= nil then
     input_layer:share(self.master_input_dssm_layer,
      'weight', 'bias', 'gradWeight', 'gradBias')
  end
  return input_layer
end

-- Returns new question embed module corresponding to config
function DSSM_Network:new_question_embed_layer(config)
  assert(config ~= nil, "Must specify config for question embed layer")
  local question_embed_layer = dmn.WordEmbedModule{
    gpu_mode = config.gpu_mode,
    emb_dim = config.question_emb_dim,
    num_classes = config.question_num_classes,
    dropout_prob = config.question_dropout_prob,
    dropout = config.question_dropout,
    hashing = self.question_vocab.hashed -- use hashing if the vocab is hashed
  }
  return question_embed_layer
end

-- Returns new input embed module corresponding to config
function DSSM_Network:new_input_embed_layer(config)
  assert(config ~= nil, "Must specify config for input embed layer")

  local input_embed_layer = dmn.WordEmbedModule{
    gpu_mode = config.gpu_mode,
    emb_dim = config.input_emb_dim,
    num_classes = config.input_num_classes,
    dropout_prob = config.input_dropout_prob,
    dropout = config.input_dropout,
    hashing = self.input_vocab.hashed -- use hashing if the vocab is hashed
  }

  if self.master_input_embed_layer ~= nil then
     dmn.logger:print(self.master_input_embed_layer.gpu_mode)
     input_embed_layer:share(self.master_input_embed_layer,
      'weight', 'bias', 'gradWeight', 'gradBias')
  end

  return input_embed_layer
end


-- enables dropouts on all layers
function DSSM_Network:enable_dropouts()
  for i = 1, #self.layers do
    dmn.logger:print("Enabling dropout on ", self.layers[i])
    self.layers[i]:enable_dropouts()
  end
end

-- disables dropouts on all layers
function DSSM_Network:disable_dropouts()
  for i = 1, #self.layers do
    dmn.logger:print("Disabling dropout on ", self.layers[i])
    self.layers[i]:disable_dropouts()
  end
end

-- Forward propagate.
-- question_indices: IntTensor which represents question indices
-- input_indices: NxIntTensor array which represents input indices for answer module.
-- returns cosine distance between two
function DSSM_Network:forward(question_indices, input_indices, corr_index)

  assert(question_indices ~= nil, "Must specify question indices to DSSM net")
  assert(input_indices ~= nil, "Must specify input indices to DSSM net")
  assert(corr_index ~= nil, "Must specify correct index for softmax")

  local t1 = sys.clock()
  self.question_emb = self.question_embed_layer:forward(question_indices)

  local t2 = sys.clock()
  self.question_dssm = self.question_dssm_layer:forward(self.question_emb)

  local t3 = sys.clock()
  self.input_dssm = self:forward_input(input_indices, false) 

  local t4 = sys.clock()
  local loss = self.answer_layer:forward(self.question_dssm, self.input_dssm, 
    corr_index) 

  local t5 = sys.clock()
  
  dmn.logger:print("Forwarding DSSM Network")
  dmn.logger:print(t5 - t4, t4 - t3, t3 - t2, t2- t1)
  return loss
end

-- returns dssm of input indices, predict or not determines whether to use same net for
-- forwarding
function DSSM_Network:forward_input(input_indices, predict)
  assert(input_indices ~= nil, "Must specify input indices to DSSM net")
  assert(predict ~= nil, "Must specify whether to predict or not")
  self.input_emb = {}
  local outputs = {}

  for i = 1, #input_indices do
    local curr_input = input_indices[i]
    local curr_network_index = predict and 1 or i
    if self.input_embed_layers[curr_network_index] == nil then
      dmn.logger:print("Creating a new input embed layer for index ", i)
      self.input_embed_layers[curr_network_index] = self:new_input_embed_layer(self.config)
    end
    if self.input_dssm_layers[curr_network_index] == nil then
      dmn.logger:print("Creating a new input dssm layer for index ", i)
      self.input_dssm_layers[curr_network_index] = self:new_input_dssm_layer(self.config)
    end

    local t1 = sys.clock()
    self.input_emb[i] = self.input_embed_layers[curr_network_index]:forward(curr_input)
    local t2 = sys.clock()
    outputs[i] = self.input_dssm_layers[curr_network_index]:forward(self.input_emb[i])
    local t3 = sys.clock()

    --print("ON forwarding inthput", t3 - t2, t2 - t1)
  end

  return outputs
end

-- Back propagate
function DSSM_Network:backward(question_indices, input_indices, best_index)
  assert(question_indices ~= nil, "Must specify question indices to DSSM net")
  assert(input_indices ~= nil, "Must specify input indices to DSSM net")
  assert(best_index ~= nil, "Must specify desired distance between question/input indices")
  
  local t1 = sys.clock()
  -- get errors wrt dssm outputs
  self.question_input_errs = self.answer_layer:backward(self.question_dssm,
                              self.input_dssm, best_index)

  local t2 = sys.clock()
  -- split that into question and input dssm errors
  self.question_dssm_errs = self.question_input_errs[1]
  self.input_dssm_errs = self.question_input_errs[2]

  -- get errors wrt to question errors
  self.question_emb_errs = self.question_dssm_layer:backward(self.question_emb, 
                            self.question_dssm_errs)

  local t3 = sys.clock()
  self.final_question_errs = self.question_embed_layer:backward(question_indices,
                              self.question_emb_errs)

  local t4 = sys.clock()
  -- get errors with respect to input indices
  self:backward_input(input_indices, self.input_dssm_errs)
  local t5 = sys.clock()

  dmn.logger:print("Backwarding DSSM network")
  dmn.logger:print(t5 - t4, t4 - t3, t3 - t2, t2- t1)
end

-- returns cosine distance between two
function DSSM_Network:backward_input(input_indices, input_errs)
  assert(input_indices ~= nil, "Must specify input indices to DSSM net")
  assert(input_errs ~= nil, "Must specify input errors to DSSM net")

  for i = 1, #input_indices do
    -- get current input
    local curr_input_indices = input_indices[i]
    local curr_input_emb = self.input_emb[i]
    local curr_input_err = input_errs[i]

    local curr_err = self.input_dssm_layers[i]:backward(curr_input_emb, curr_input_err)
    local input_err = self.input_embed_layers[i]:backward(curr_input_indices, curr_err)
  end
end

-- Predicts most likely candidates for each question in the dataset
-- dataset: dataset to predict on
-- beam_size: search size for prediction. If -1 then return all items
-- num_predictions: number of items in dataset to predict on
function DSSM_Network:predict_dataset(dataset, beam_size, num_predictions)
  assert(dataset ~= nil, "Must specify dataset to predict in")
  assert(beam_size ~= nil, "Must specify beam size to use for predictions")
  assert(num_predictions ~= nil, "Must specify number of predictions to predict")
  assert(num_predictions <= dataset.size, "Number of predictions must not exceed dataset size")
 
  self:disable_dropouts()

  local beam_size = beam_size
  local predictions = {}
  for i = 1, num_predictions do
    if i % 100 == 0 then
      collectgarbage()
    end
    xlua.progress(i, num_predictions)
    local question_indices = dataset.questions[i]

    -- Forward the candidates
    local candidate_inputs = #dataset.candidate_inputs == 1 and dataset.candidate_inputs[1] or dataset.candidate_inputs[i]
    local cur_beam_size = (beam_size == -1) and #candidate_inputs or beam_size

    prediction = self:predict_tokenized(question_indices, candidate_inputs, cur_beam_size)
    assert(cur_beam_size == #prediction, "Beam size must equal the number of predictions")
    table.insert(predictions, dmn.functions.deepcopy(prediction))
  end

  for i = 1, num_predictions do 
    if beam_size == -1 then
      assert(#dataset.candidate_inputs[i] == #predictions[i], "Must rerank all inputs")
    end
  end

  return predictions
end

-- Predicts for a given question, candidate inputs and beam size
-- the most likely candidates. 
-- Returns a list with the candidates, probabilities and rankings
-- question: Sentence of question
-- candidate_inputs: Table of inputs
-- beam_size: Size of inputs
function DSSM_Network:predict(question, candidate_inputs, beam_size)
  assert(question ~= nil, "Must specify question to predict")
  assert(candidate_inputs ~= nil, "Must specify input sentences to forward")
  assert(beam_size ~= nil, "Must specify beam size to predict")

  local cur_beam_size = (beam_size == -1) and #candidate_inputs or beam_size

  function get_tokens(sent, vocab)
    local indeces = datasets.get_input_tokens(sent, vocab, self.gpu_mode, self.char_level, self.pad_size)
    return indeces
  end

  function get_multi_tokens(sentences, vocab)
    local tokens = {}
    local gpu_mode = false
    for i = 1, #sentences do
      local curr_tokens = get_tokens(sentences[i], vocab, gpu_mode, self.char_level, self.pad_size)
      table.insert(tokens, curr_tokens)
    end
    return tokens
  end

  local question_tokens = get_tokens(question, self.question_vocab)
  local input_tokens = get_multi_tokens(candidate_inputs, self.input_vocab) 

  assert(#input_tokens ~= 0, "Must give nonzero number of predicted tokens")

  local predicted_list = self:predict_tokenized(question_tokens, input_tokens, cur_beam_size)
  
  assert(#predicted_list ~= 0, "Must have at least 1 prediction index")

  -- TODO NOT SURE HOW TO RECONSTRUCT SENTENCE FOR TOKENIZED INPUT
  local best_indices = {}
  local likelihoods = {}

  for i = 1, #predicted_list do 
    local best_index = predicted_list[i][3]
    local likelihood = predicted_list[i][1]
    table.insert(best_indices, best_index)
    table.insert(likelihoods, likelihood)
  end
  
  return best_indices, likelihoods
end

-- predicts with tokenized question indices and word indices
function DSSM_Network:predict_tokenized(question_indices, input_indices_list, beam_size)
  assert(question_indices ~= nil, "Must specify question indices to predict")
  assert(input_indices_list ~= nil, "Must specify input indices to predict")
  assert(beam_size ~= nil, "Must specify beam size to use for decoding")

  local t1 = sys.clock()
  local question_emb = self.question_embed_layer:forward(question_indices)

  local t2 = sys.clock()
  local question_dssm = self.question_dssm_layer:forward(question_emb)

  local t3 = sys.clock()
  local input_dssm = self:forward_input(input_indices_list, false) 

  local t4 = sys.clock()
  assert(#input_dssm == #input_indices_list, "DSSM output must have same size as input")

  -- now get predictions from best indices
  local prediction_indices = self.answer_layer:predict(question_dssm, input_dssm, beam_size)
  local t5 = sys.clock()

  --print("Predicting DSSM network")
  --print(t5 - t4, t4 - t3, t3 - t2, t2- t1)
  local predictions = {}

  for i = 1, #prediction_indices do
    local curr_index = prediction_indices[i][2]
    local likelihood = prediction_indices[i][1]
    table.insert(predictions, {likelihood, input_indices_list[curr_index], curr_index})
  end

  return predictions
end

-- test predictions: predictions to get sentences from
function DSSM_Network:get_sentences(test_predictions, candidates)
  dmn.logger:print("Getting sentences")
  local sentences = {}
  --predictions_file:write("LOSS " .. loss .. '\n')
  for i = 1, #test_predictions do
    local test_prediction = test_predictions[i]
    for j = 1, 1 do
      local test_prediction = test_predictions[i][j]
      local sentence = self:get_sentence(test_prediction, candidates)
      table.insert(sentences, sentence)
    end
  end
  return sentences
end

-- gets sentence for single test prediction
function DSSM_Network:get_sentence(test_prediction, candidates)
    local likelihood = test_prediction[1]
    local tokens = test_prediction[2]
    local index = test_prediction[3]

    -- two cases, if embeddings then can reconstruct input vocab
    -- Remove tokens
    if self.input_vocab.hashed then
      dmn.logger:print("Getting hashed rep (need original list of candidates")
      assert(candidates ~= nil, "Must specify candidates if using hash vocab")
      local candidate = candidates[index]
      return candidate
    else 
      local sentence = table.concat(self.input_vocab:tokens(tokens), ' ')
      local sentence = string.gsub(sentence, "</s>", "")
      local sentence = string.gsub(sentence, "<s>", "")
      return sentence
    end
end

function DSSM_Network:sample_inputs(inputs, index_to_avoid, num_inputs)
  assert(inputs ~= nil, "Dataset to evaluate must not be null")
  assert(index_to_avoid ~= nil, "Must specify the index to avoid")
  assert(num_inputs ~= nil, "Must specify number of inputs to the dataset")

  local neg_sample = datasets.generate_negative_sample(inputs, inputs[index_to_avoid], num_inputs)
  assert(#neg_sample == num_inputs, "Number of negative samples must equal number of inputs specified")
  return neg_sample
end

function DSSM_Network:eval(dataset)
  assert(dataset ~= nil, "Dataset to evaluate must not be null")

  local num_correct = 0
  local num_total = dataset.size

  -- first get predictions from dataset
  local all_predictions = self:predict_dataset(dataset, 3, dataset.size)  
  for i = 1, #all_predictions do
    -- concatenate all inputs. Positive input goes first, then negative inputs
    local new_inputs = {}
    local raw_positive_inputs = dataset.raw_inputs[i]
    local curr_predictions = all_predictions[i]
    local prediction_index = curr_predictions[1][3]
    local question_raw = dataset.raw_questions[i]
    local prediction_raw = dataset.raw_candidate_inputs[prediction_index]
    local desired_raw = dataset.raw_inputs[i]

    if i % 100 == 0 then
      collectgarbage()
    end
    dmn.logger:print("Predictions")
    dmn.logger:print(question_raw)
    dmn.logger:print(prediction_raw)
    dmn.logger:print(desired_raw)
    if prediction_raw == desired_raw then
      dmn.logger:print("CORRECT")
      num_correct = num_correct + 1
    end
  end

  local accuracy_at_1 = num_correct / num_total
  return accuracy_at_1
end

-- helper function to do forward backward
-- question_indices: indices of question to forward/backward
-- input_indices: indices of input to forward/backward
-- expected_similarity: expected_similarity to forward/backward
-- returns RMSE between similarity and desired similarity
function DSSM_Network:forward_backward(question_indices, input_indices, expected_similarity)
  assert(question_indices ~= nil, "Must specify question indices to forward/backward")
  assert(input_indices ~= nil, "Must specify input indices to forward/backward")
  assert(expected_similarity ~= nil, "Must specify loss to forward/backward")

  local loss = self:forward(question_indices, input_indices, expected_similarity)
  self:backward(question_indices, input_indices, expected_similarity)
  
  return loss
end

function DSSM_Network:train(dataset)
  assert(dataset ~= nil, "Dataset to evaluate must not be null")
  self:enable_dropouts()

  local indices = torch.randperm(dataset.size)
  local tot_loss = 0
  local tot_num_items = 0
  local num_negative_samples = 30
  local cur_collect_garbage = 1
  
  --dataset.size
  for i = 1, dataset.size, self.batch_size do --dataset.size, self.batch_size do
    xlua.progress(i, dataset.size)
    local batch_size = math.min(i + self.batch_size - 1, dataset.size) - i + 1
    currIndex = 0

    if i - cur_collect_garbage > 2000 then 
      collectgarbage()
      dmn.logger:print("Collecting garbage, current free memory is " .. collectgarbage("count")*1024)
      cur_collect_garbage = i
    end

    local feval = function(x)
      self.grad_params:zero()
      local start = sys.clock()
      local loss = 0
      local num_items = 0
      for j = 1, batch_size do
        local idx = indices[i + j - 1]
        local desired_index = 1

        -- first forward/backward positive samples
        local question_indices = dataset.questions[idx]
        local positive_inputs = dataset.positive_inputs[idx]
        local negative_input_indices = self:sample_inputs(dataset.positive_inputs, idx, num_negative_samples)

        -- concatenate all inputs. Positive input goes first, then negative inputs
        local new_inputs = {positive_inputs}

        for k = 1, #negative_input_indices do
          local curr_neg_index = negative_input_indices[k]
          new_inputs[k + 1] = dataset.positive_inputs[curr_neg_index]
        end

        local curr_loss = self:forward_backward(question_indices, 
                                               new_inputs, 
                                               desired_index)
        loss = loss + curr_loss
      end

      tot_loss = tot_loss + loss
      tot_num_items = tot_num_items + batch_size
      loss = loss / batch_size
      self.grad_params:div(batch_size)

      -- clamp grad params in accordance to karpathy from -5 to 5
      --self.grad_params:clamp(-5, 5)
      -- regularization: BAD BAD BAD
      --loss = loss + 0.5 * self.reg * self.params:norm() ^ 2
      --self.grad_params:add(self.reg, self.params)
      dmn.logger:print("Current loss", loss)
      --print(currIndex, "of", self.params:size())
      currIndex = currIndex + 1
      return loss, self.grad_params
    end
    -- check gradients for lstm layer
    --diff, DC, DC_est = optim.checkgrad(feval, self.params, 1e-7)
    --print("Gradient error for dssm network  is")
    --print(diff)
    --assert(diff < 1e-5, "Gradient is greater than tolerance")
    --assert(false)
    self.optim_method(feval, self.params, self.optim_state)
  end


  average_loss = tot_loss / tot_num_items
  xlua.progress(dataset.size, dataset.size)
  return average_loss
end

-- Forgets depths for all layers. Important so that we don't create new cells infinitely
function DSSM_Network:forget()
  for i = 1, #self.layers do
    self.layers[i]:forget()
  end
end

-- saves prediction to specified file path
function DSSM_Network:save_predictions(predictions_save_path, loss, test_predictions, raw_candidates)
  assert(predictions_save_path ~= nil, "Must specify predictions save path to save to")
  assert(loss ~= nil, "Must specify loss for predictions")
  assert(test_predictions ~= nil, "Must specify test predictions")
  assert(raw_candidates ~= nil, "Must specify raw candidates for reconstructing predictions")
  local predictions_file, err = io.open(predictions_save_path,"w")

  dmn.logger:print('writing predictions to ' .. predictions_save_path)
  --predictions_file:write("LOSS " .. loss .. '\n')
  local sentences = self:get_sentences(test_predictions, raw_candidates)
  for i = 1, #sentences do
    local sentence = sentences[i]
    if i < #sentences then 
      predictions_file:write(sentence .. '\n')
    else
      predictions_file:write(sentence)
    end
  end
  predictions_file:close()
end

function DSSM_Network:print_config()
  local num_params = self.params:size(1)
  dmn.logger:printf('%-25s = %d\n', 'num params', num_params)
  dmn.logger:printf('%-25s = %d\n', 'batch size', self.batch_size)
  dmn.logger:printf('%-25s = %s\n', 'optim_method', self.optim_method_string)
  dmn.logger:printf('%-25s = %s\n', 'optim_state', self.optim_state.learningRate)
  dmn.logger:printf('%-25s = %d\n', 'model epoch', self.model_epoch)
  for i = 1, #self.layers do
    dmn.logger:printf("\n\n===== dmn.logger:printing config for layer %s =====\n\n", self.layers[i])
    self.layers[i]:print_config()
  end
end

function DSSM_Network:get_path(index)
  return "DSSM_Network_" .. index .. ".th"
end

function DSSM_Network:save(path, model_epoch)
  local config = self.config
  config.model_epoch = (model_epoch ~= nil) and model_epoch or self.model_epoch
  torch.save(path, {
    params = self.params,
    config = config,
    optim_state = self.optim_state
  })
end

function DSSM_Network.load(path)
  local state = torch.load(path)
  local model = dmn.DSSM_Network.new(state.config)
  model.params:copy(state.params)
  return model
end

