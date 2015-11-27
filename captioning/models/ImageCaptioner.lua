--[[

  Image Captioning using an LSTM and a softmax for the language model

--]]

local ImageCaptioner = torch.class('imagelstm.ImageCaptioner')

function ImageCaptioner:__init(config)
  self.gpu_mode                = config.gpu_mode          or false
  self.reverse                 = config.reverse           or true
  self.image_dim               = config.image_dim         or 1024
  self.combine_module_type     = config.combine_module    or "embedlayer"
  self.hidden_module_type      = config.hidden_module     or "projlayer"
  self.mem_dim                 = config.mem_dim           or 150
  self.emb_dim                 = config.emb_dim           or 50
  self.learning_rate           = config.learning_rate     or 0.01
  self.batch_size              = config.batch_size        or 100
  self.emb_vecs                = config.emb_vecs          
  self.dropout                 = (config.dropout == nil) and false or config.dropout
  self.optim_method            = config.optim_method or optim.rmsprop
  self.num_classes             = config.num_classes or 2944
  self.optim_state             = config.optim_state or 
                                {learning_rate = self.learning_rate,
                                   learningRateDecay = 0,
                                   weightDecay = 0,
                                   --momentum = 0.1
                                }
  self.num_layers              = config.num_layers or 1
  self.vocab                   = config.vocab
  self.in_dropout_prob         = config.in_dropout_prob or 0.5
  self.hidden_dropout_prob     = config.hidden_dropout_prob or 0.5
  self.emb_vecs                = config.emb_vecs
  self.model_epoch             = config.model_epoch or -1
  if config.emb_vecs ~= nil then
    self.num_classes = config.emb_vecs:size(1)
  end

  self.combine_layer = self:get_combine_layer(self.combine_module_type)
  
  self.hidden_layer = self:get_hidden_layer(self.hidden_module_type)
 
  self.in_zeros = torch.zeros(self.emb_dim)
  self.optim_state = { learningRate = self.learning_rate }
  
  -- negative log likelihood optimization objective
  local num_caption_params = self:new_caption_module():getParameters():size(1)
  print("Number of caption parameters " .. num_caption_params)

  self.image_captioner = imagelstm.ImageCaptionerLSTM{
    gpu_mode = self.gpu_mode,
    in_dim  = self.combine_layer:getOutputSize(),
    mem_dim = self.mem_dim,
    num_layers = self.num_layers,
    output_module_fn = self:new_caption_module(),
    criterion = nn.ClassNLLCriterion()
  }


  -- set gpu mode
  if self.gpu_mode then
    self:set_gpu_mode()
  end
  
  local captioner_modules = self.image_captioner:getModules()
  local combine_modules = self.combine_layer:getModules()
  local hidden_modules = self.hidden_layer:getModules()

  local modules = nn.Parallel()
  self:add_modules(modules, captioner_modules)
  self:add_modules(modules, combine_modules)
  self:add_modules(modules, hidden_modules)

  print("==== Modules we're optimizing === ")
  print(modules)
  self.params, self.grad_params = modules:getParameters()
end

-- Returns input layer module corresponding to type specified by parameter
-- Combine_module_type: type of input layer to return
-- Requires: combine_module_type not nil, one of "addlayer", "concatlayer",
-- "singleaddlayer", "concatprojlayer", "embedlayer"
-- Returns: combine layer module corresponding to this layer
function ImageCaptioner:get_combine_layer(combine_module_type)
  assert(combine_module_type ~= nil)
  local layer
  if combine_module_type == "addlayer" then
      layer = imagelstm.AddLayer{
      emb_dim = self.emb_dim,
      num_classes = self.num_classes,
      gpu_mode = self.gpu_mode,
      dropout = self.dropout,
      dropout_prob = self.in_dropout_prob,
      image_dim = self.image_dim
    }
  elseif combine_module_type == "concatlayer" then
    layer = imagelstm.ConcatLayer{
      emb_dim = self.emb_dim,
      num_classes = self.num_classes,
      gpu_mode = self.gpu_mode,
      dropout = self.dropout,
      dropout_prob = self.in_dropout_prob,
      image_dim = self.image_dim  
    }
  elseif combine_module_type == "singleaddlayer" then
    layer = imagelstm.SingleAddLayer{
      emb_dim = self.emb_dim,
      num_classes = self.num_classes,
      gpu_mode = self.gpu_mode,
      dropout = self.dropout,
      dropout_prob = self.in_dropout_prob,
      image_dim = self.image_dim
    }
  elseif combine_module_type == "concatprojlayer" then
    layer = imagelstm.ConcatProjLayer{
      emb_dim = self.emb_dim,
      num_classes = self.num_classes,
      gpu_mode = self.gpu_mode,
      dropout = self.dropout,
      dropout_prob = self.in_dropout_prob,
      image_dim = self.image_dim
    }
  elseif combine_module_type == "embedlayer" then
    layer = imagelstm.EmbedLayer{  
    emb_dim = self.emb_dim,
    num_classes = self.num_classes,
    gpu_mode = self.gpu_mode,
    dropout = self.dropout,
    emb_vecs = self.emb_vecs,
    dropout_prob = self.in_dropout_prob,
    image_dim = self.image_dim
    }
  else -- module not recognized
    error("Did not recognize input module type", combine_module_type)
  end
  return layer
end

-- Returns hidden layer module corresponding to type specified by parameter
-- Hidden_module_type: type of input layer to return
-- Requires: hidden_module_type not nil, one of "projlayer"
-- Returns: hidden layer module corresponding to this layer
function ImageCaptioner:get_hidden_layer(hidden_module_type)
  assert(hidden_module_type ~= nil)
  local layer
  if hidden_module_type == "projlayer" then
    layer = imagelstm.HiddenProjLayer{
      gpu_mode = self.gpu_mode,
      image_dim = self.image_dim,
      mem_dim = self.mem_dim,
      num_layers = self.num_layers,
      dropout = self.dropout,
      dropout_prob = self.hidden_dropout_prob
    }
  elseif hidden_module_type == "singleprojlayer" then
    layer = imagelstm.HiddenSingleProjLayer{
      gpu_mode = self.gpu_mode,
      image_dim = self.image_dim,
      mem_dim = self.mem_dim,
      num_layers = self.num_layers,
      dropout = self.dropout,
      dropout_prob = self.hidden_dropout_prob
    }
  else
    layer = imagelstm.HiddenDummyLayer{
      gpu_mode = self.gpu_mode,
      image_dim = self.image_dim,
      mem_dim = self.mem_dim,
      num_layers = self.num_layers,
      dropout = self.dropout,
      dropout_prob = self.hidden_dropout_prob
    }
  end
  return layer
end
-- adds modules into parallel network from module list
-- requires parallel_net is of type nn.parallel
-- requires module_list is an array of modules that is not null
-- modifies: parallel_net by adding modules into parallel net
function ImageCaptioner:add_modules(parallel_net, module_list)
  assert(parallel_net ~= nil)
  assert(module_list ~= nil)
  for i = 1, #module_list do
    curr_module = module_list[i]
    parallel_net:add(curr_module)
  end
end

-- Set all of the network parameters to gpu mode
function ImageCaptioner:set_gpu_mode()
  self.gpu_mode = true
  self.image_captioner:set_gpu_mode()
  self.combine_layer:set_gpu_mode()
  self.hidden_layer:set_gpu_mode()
end

function ImageCaptioner:set_cpu_mode()
  self.gpu_mode = false
  self.image_captioner:set_cpu_mode()
  self.combine_layer:set_cpu_mode()
  self.hidden_layer:set_cpu_mode()
end

function ImageCaptioner:new_caption_module()
  local caption_module = nn.Sequential()
  if self.dropout then
    caption_module:add(nn.Dropout(0.5, false))
  end
  caption_module
    :add(nn.Linear(self.mem_dim, self.num_classes))
    :add(nn.LogSoftMax())
  return caption_module
end

-- enables dropouts on all layers
function ImageCaptioner:enable_dropouts()
  self.image_captioner:enable_dropouts()
  self.combine_layer:enable_dropouts()
  self.hidden_layer:enable_dropouts()
end

-- disables dropouts on all layers
function ImageCaptioner:disable_dropouts()
  self.image_captioner:disable_dropouts()
  self.combine_layer:disable_dropouts()
  self.hidden_layer:disable_dropouts()
end

function ImageCaptioner:train(dataset)
  assert(dataset ~= nil)
  self:enable_dropouts()

  local indices = torch.randperm(dataset.size)
  local zeros = torch.zeros(self.mem_dim)
  local tot_loss = 0
  --dataset.size
  for i = 1, dataset.size, self.batch_size do --dataset.size, self.batch_size do
    xlua.progress(i, dataset.size)
    local batch_size = math.min(i + self.batch_size - 1, dataset.size) - i + 1
    
    currIndex = 0
    local feval = function(x)
      self.grad_params:zero()
      local start = sys.clock()
      local loss = 0
      for j = 1, batch_size do
        local idx = indices[i + j - 1]
        
        local idx = i + j - 1
        -- get the image features
        local imgid = dataset.image_ids[idx]
        local image_feats = dataset.image_feats[imgid]

        -- get input and output sentences
        local sentence = dataset.sentences[idx]
        local out_sentence = dataset.pred_sentences[idx]

        -- get text/image inputs
        local inputs = self.combine_layer:forward(sentence, image_feats)
        local hidden_inputs = self.hidden_layer:forward(image_feats)


        local lstm_output, class_predictions, caption_loss = 
        self.image_captioner:forward(inputs, hidden_inputs, out_sentence)
        
        loss = loss + caption_loss

        local input_grads, hidden_grads = 
        self.image_captioner:backward(inputs, hidden_inputs, lstm_output, class_predictions, out_sentence)
        
        -- do backward through input to lstm
        self.hidden_layer:backward(image_feats, hidden_grads)
        self.combine_layer:backward(sentence, image_feats, input_grads)
      end

      tot_loss = tot_loss + loss
      loss = loss / batch_size
      self.grad_params:div(batch_size)

      -- clamp grad params in accordance to karpathy from -5 to 5
      self.grad_params:clamp(-5, 5)
      -- regularization: BAD BAD BAD
      -- loss = loss + 0.5 * self.reg * self.params:norm() ^ 2
      -- self.grad_params:add(self.reg, self.params)
      -- print("Current loss", loss)
      -- print(currIndex, " of ", self.params:size(1))
      currIndex = currIndex + 1
      return loss, self.grad_params
    end
    -- check gradients for lstm layer
    -- diff, DC, DC_est = optim.checkgrad(feval, self.params, 1e-7)
    -- print("Gradient error for lstm captioner is")
    -- print(diff)
    -- assert(diff < 1e-5, "Gradient is greater than tolerance")

    self.optim_method(feval, self.params, self.optim_state)
  end
  average_loss = tot_loss / dataset.size
  xlua.progress(dataset.size, dataset.size)

  return average_loss
end

-- Evaluates model on dataset
-- Returns average loss
function ImageCaptioner:eval(dataset)
  assert(dataset ~= nil)
  self:disable_dropouts()

  local indices = torch.randperm(dataset.size)
  local zeros = torch.zeros(self.mem_dim)
  local tot_loss = 0
  local tot_ppl2 = 0
  local num_words = 0
  --dataset.size
  for i = 1, dataset.size, self.batch_size do --dataset.size, self.batch_size do
    xlua.progress(i, dataset.size)
    local batch_size = math.min(i + self.batch_size - 1, dataset.size) - i + 1
    
    currIndex = 0
    local feval = function(x)
      local start = sys.clock()
      local loss = 0
      for j = 1, batch_size do
        self.image_captioner:reset_depth()
        local idx = indices[i + j - 1]
        
        --local idx = i + j - 1
        -- get the image features
        local imgid = dataset.image_ids[idx]
        local image_feats = dataset.image_feats[imgid]

        -- get input and output sentences
        local sentence = dataset.sentences[idx]

        local out_sentence = dataset.pred_sentences[idx]

        -- get text/image inputs
        local inputs = self.combine_layer:forward(sentence, image_feats)
        local hidden_inputs = self.hidden_layer:forward(image_feats)

        local lstm_output, class_predictions, caption_loss = 
        self.image_captioner:forward(inputs, hidden_inputs, out_sentence)
        
        loss = loss + caption_loss

        local sent_perp = 0.0
        for k = 1, out_sentence:size(1) do
          sent_perp = sent_perp - class_predictions[k][out_sentence[k]]
        end
        tot_ppl2 = tot_ppl2 + sent_perp
        num_words = num_words + sentence:size(1) + 0.0
      end

      tot_loss = tot_loss + loss

      loss = loss / batch_size
      self.grad_params:div(batch_size)

      -- regularization: BAD BAD BAD
      -- loss = loss + 0.5 * self.reg * self.params:norm() ^ 2
      -- self.grad_params:add(self.reg, self.params)
      -- print(currIndex, " of ", self.params:size(1))
      --currIndex = currIndex + 1
      return loss, self.grad_params
    end
    feval()
  end
  local average_loss = tot_loss / dataset.size

  -- perplexity is 2**log_2(sum(losses)) / dataset.size
  local norm_ppl = tot_ppl2 / num_words
  local perplexity = math.exp(norm_ppl)
  xlua.progress(dataset.size, dataset.size)
  return average_loss, perplexity
end

function ImageCaptioner:predict(image_features, beam_size)
  assert(image_features ~= nil)
  --assert(beam_size > 0, "Beam size must be a positive number")

  -- Keep track of tokens predicted
  local num_iter = 0
  local tokens = {}

  -- does one tick of lstm, input token and previous lstm state as input
  -- next_token: input into lstm
  -- curr_iter: current iteration
  -- prev_outputs: hidden state, cell state of lstm
  -- returns predicted token, its log likelihood, state of lstm, and all predictions

  local function lstm_tick(next_token, prev_outputs, curr_iter)
   assert(next_token ~= nil)
   assert(prev_outputs ~= nil)

   local start_time = sys.clock()
   local inputs = self.combine_layer:forward(next_token, image_features, curr_iter)

   local t1 = sys.clock()
   -- feed forward to predictions
   local next_outputs, class_predictions = self.image_captioner:tick(inputs[1], prev_outputs)

   local t2 = sys.clock()
   --print(class_predictions:size())
   --local squeezed_predictions = torch.squeeze(class_predictions)
   --print(squeezed_predictions:size())
   local t3 = sys.clock()
   local predicted_token = argmax(class_predictions, num_iter < 3)
   
   local t4 = sys.clock()
   --print("Predicted token ")
   --print(predicted_token, next_token)
   local likelihood = class_predictions[predicted_token]

   --print("=== TIMES ===")
   --print("Tot time", t4 - t1, t4 - t3, t3 - t2, t2 - t1)
   return predicted_token, likelihood, next_outputs, class_predictions
  end

  -- Start with special START token:
  local next_token = self.gpu_mode and torch.CudaTensor{self.vocab.start_index} 
                      or torch.IntTensor{self.vocab.start_index}

  -- Terminate when predict the END token
  local end_token = self.gpu_mode and torch.CudaTensor{self.vocab.end_index} 
                      or torch.IntTensor{self.vocab.end_index}

  -- Initial hidden state/cell state values for lstm
  local prev_outputs = self.hidden_layer:forward(image_features)

  if beam_size < 2 then
    local ll = 0
    -- Greedy search
    while next_token[1] ~= end_token[1] and num_iter < 20 do
       -- get text/image inputs
      local pred_token, likelihood, next_outputs, class_predictions = 
                        lstm_tick(next_token, prev_outputs, num_iter)
     
      -- keep count of number of tokens seen already
      num_iter = num_iter + 1
      ll = ll + likelihood
      if pred_token ~= end_token then
        table.insert(tokens, pred_token)
      end

      -- convert token into proper format for feed-forwarding
      next_token[1] = pred_token
      prev_outputs = self:copy(next_outputs)
    end
    return {{ll, tokens}}
  else
    -- beam search
    local beams = {}

    -- first get initial predictions
    local pred_token, likelihood, next_outputs, class_predictions = lstm_tick(next_token, prev_outputs, num_iter)
    

    -- then get best top k indices indices
    local best_indices = topkargmax(class_predictions, beam_size)

    -- then initialize our tokens list
    for i = 1, beam_size do
      local next_token = best_indices[i]
      local copied_outputs = self:copy(next_outputs)
      --local copied_outputs = next_outputs
      local curr_beam = {class_predictions[next_token], {next_token}, copied_outputs}
      table.insert(beams, curr_beam)
    end

    num_iter = num_iter + 1
    -- now do the general beam search algorithm
    while num_iter < 20 do
      local next_beams = {}

      for i = 1, #beams do
        local curr_beam = beams[i]

        -- get all predicted tokens so far
        local curr_tokens_list = curr_beam[2]
        local next_token = self.gpu_mode and 
                          torch.CudaTensor{curr_tokens_list[#curr_tokens_list]} or
                          torch.IntTensor{curr_tokens_list[#curr_tokens_list]}
        -- If the next token is the end token, just add prediction to list
        if next_token[1] == end_token[1] then
           if num_iter > 5 then 
              table.insert(next_beams, curr_beam)
           end
        else 
          local log_likelihood = curr_beam[1]
          local prev_output = curr_beam[3]

          -- first get initial predictions
          local pred_token, likelihood, next_out, class_pred = 
          lstm_tick(next_token, prev_output, num_iter)

          -- then get best top k indices indices
          local best_indices = topkargmax(class_predictions, beam_size)
      -- then initialize our tokens list
          for i = 1, beam_size do
            local next_tokens_list = {}
            -- copy current tokens over
            for j = 1, #curr_tokens_list do
              table.insert(next_tokens_list, curr_tokens_list[j])
            end
            local next_token = best_indices[i]
            table.insert(next_tokens_list, next_token)

            local next_ll = log_likelihood + class_pred[next_token]
            local copied_next_out = self:copy(next_out)
            --local copied_next_out = next_out
            local next_beam = {next_ll, next_tokens_list, copied_next_out}

            table.insert(next_beams, next_beam)
          end
        end
      end

      num_iter = num_iter + 1
      -- Keep top beam_size entries in beams
      beams = topk(next_beams, beam_size)
    end
    return beams
  end
end

function ImageCaptioner:copy(prev_outputs)
  local copied_prev_outputs = {}
  if self.num_layers == 1 then 
    local first_input = self.gpu_mode and 
                 torch.CudaTensor(prev_outputs[1]:size()):copy(prev_outputs[1])
                 or torch.Tensor(prev_outputs[1]:size()):copy(prev_outputs[1])
    local second_input = self.gpu_mode and 
              torch.CudaTensor(prev_outputs[2]:size()):copy(prev_outputs[2])
              or torch.Tensor(prev_outputs[2]:size()):copy(prev_outputs[2])

    if self.gpu_mode then
      first_input = first_input:cuda()
      second_input = second_input:cuda()
    end
    
    table.insert(copied_prev_outputs, first_input)
    table.insert(copied_prev_outputs, second_input)
  else 
    local curr_cells = {}
    local curr_hiddens = {}
    for i = 1, self.num_layers do 
      local curr_outputs = {}
      local first_input = self.gpu_mode and 
                 torch.CudaTensor(prev_outputs[1][i]:size()):copy(prev_outputs[1][i])
                 or torch.Tensor(prev_outputs[1][i]:size()):copy(prev_outputs[1][i])
      local second_input = self.gpu_mode and 
              torch.CudaTensor(prev_outputs[2][i]:size()):copy(prev_outputs[2][i])
              or torch.Tensor(prev_outputs[2][i]:size()):copy(prev_outputs[2][i])

      if self.gpu_mode then
        first_input = first_input:cuda()
        second_input = second_input:cuda()
      end
      
      table.insert(curr_cells, first_input)
      table.insert(curr_hiddens, second_input)
    end

    table.insert(copied_prev_outputs, curr_cells)
    table.insert(copied_prev_outputs, curr_hiddens)
  end
  return copied_prev_outputs
end

function ImageCaptioner:predict_dataset(dataset, beam_size, num_predictions)
  self:disable_dropouts()
  local beam_size = beam_size or 1
  local predictions = {}
  for i = 1, num_predictions do
    xlua.progress(i, num_predictions)
    local imgid = dataset.single_image_ids[i]
    local image_feats = dataset.image_feats[imgid]
    prediction = self:predict(image_feats, beam_size)
    table.insert(predictions, prediction)
  end
  return predictions
end

-- saves prediction to specified file path
function ImageCaptioner:save_predictions(predictions_save_path, loss, test_predictions)
  local predictions_file, err = io.open(predictions_save_path,"w")

  print('writing predictions to ' .. predictions_save_path)
  --predictions_file:write("LOSS " .. loss .. '\n')
  local sentences = self:get_sentences(test_predictions)
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

-- gets sentences from predictions
-- test predictions: predictions to get sentences from
function ImageCaptioner:get_sentences(test_predictions)
  local sentences = {}
  --predictions_file:write("LOSS " .. loss .. '\n')
  for i = 1, #test_predictions do
    local test_prediction = test_predictions[i]
    for j = 1, 1 do
      local test_prediction = test_predictions[i][j]
      local likelihood = test_prediction[1]
      local tokens = test_prediction[2]
      -- Remove tokens
      local sentence = table.concat(self.vocab:tokens(tokens), ' ')
      local sentence = string.gsub(sentence, "</s>", "")
      table.insert(sentences, sentence)
    end
  end
  return sentences
end

function ImageCaptioner:print_config()
  local num_params = self.params:size(1)
  local num_caption_params = self:new_caption_module():getParameters():size(1)
  printf('%-25s = %d\n', 'num params', num_params)
  printf('%-25s = %d\n', 'num compositional params', num_params - num_caption_params)
  printf('%-25s = %d\n', 'word vector dim', self.emb_dim)
  printf('%-25s = %d\n', 'LSTM memory dim', self.mem_dim)
  printf('%-25s = %d\n', 'Model epoch', self.model_epoch)
  printf('%-25s = %d\n', 'minibatch size', self.batch_size)
  printf('%-25s = %.2e\n', 'learning rate', self.learning_rate)
  printf('%-25s = %.2e\n', 'input layer dropout prob', self.in_dropout_prob)
  printf('%-25s = %.2e\n', 'hidden layer dropout prob', self.hidden_dropout_prob)
  printf('%-25s = %s\n', 'optim method', self.optim_method)
  printf('%-25s = %d\n', 'number of classes', self.num_classes)
  printf('%-25s = %d\n', 'number of layers in lstm', self.num_layers)
  printf('%-25s = %s\n', 'combine module type', self.combine_module_type)
  printf('%-25s = %s\n', 'hidden module type', self.hidden_module_type)
  printf('%-25s = %s\n', 'dropout', tostring(self.dropout))
end

function ImageCaptioner:save(path, epoch)

  local config = {
    reverse           = self.reverse,
    batch_size        = self.batch_size,
    dropout           = self.dropout,
    num_classes       = self.num_classes,
    learning_rate     = self.learning_rate,
    mem_dim           = self.mem_dim,
    reg               = self.reg,
    emb_dim           = self.emb_dim,
    emb_vecs          = self.emb_vecs,
    optim_method      = self.optim_method,
    combine_module    = self.combine_module_type,
    hidden_module     = self.hidden_module_type,
    num_layers        = self.num_layers,
    vocab             = self.vocab,
    in_dropout_prob   = self.in_dropout_prob,
    hidden_dropout_prob = self.hidden_dropout_prob,
    model_epoch = epoch or -1
  }

  local params = self.params
  local optim_state = {learning_rate = self.learning_rate,
                                   learningRateDecay = 0,
                                   weightDecay = 0,
                                   --momentum = 0.1
                           }

  if self.gpu_mode then 
    params = self.params:double()
  end

  torch.save(path, {
    params = params,
    hidden_params = self.hidden_layer:getWeights():double(),
    combine_params = self.combine_layer:getWeights():double(),
    caption_params = self.image_captioner:getWeights():double(),
    optim_state = optim_state,
    config = config,
  })
end

-- returns model path representation based on the model configuration
-- epoch: model epoch to return
function ImageCaptioner:getPath(epoch) 
  local model_save_path = string.format(
  '/image_captioning_lstm.hidden_type_%s.input_type_%s.emb_dim_%d.num_layers_%d.mem_dim_%d.epoch_%d.th', 
  self.hidden_module_type,
  self.combine_module_type,
  self.emb_dim, self.num_layers,
  self.mem_dim, epoch)
  return model_save_path
end

function ImageCaptioner.load(path)
  local state = torch.load(path)
  local model = imagelstm.ImageCaptioner.new(state.config)
  
  model.params:copy(state.params)
  if model.hidden_layer.params ~= nil then
    model.hidden_layer.params:copy(state.hidden_params)
  end
  model.combine_layer.params:copy(state.combine_params)
  model.image_captioner.params:copy(state.caption_params)
  model.optim_state = state.optim_state

  return model
end
