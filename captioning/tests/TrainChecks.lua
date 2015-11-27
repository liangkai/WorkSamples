--[[

  Image Captioning using an LSTM and a softmax for the language model

--]]

local TrainChecks = torch.class('imagelstm.TrainChecks')

function TrainChecks:__init(config)
 -- directory containing dataset files
  local data_dir = 'data/flickr8k/'

  -- load vocab
  local vocab = imagelstm.Vocab(data_dir .. 'vocab.txt')

  -- load embeddings
  print('Loading word embeddings')
  local emb_dir = 'data/glove/'
  local emb_prefix = emb_dir .. 'glove.840B'
  local emb_vocab, emb_vecs = imagelstm.read_embedding(emb_prefix .. '.vocab', emb_prefix .. '.300d.th')
  local emb_dim = emb_vecs:size(2)

  -- use only vectors in vocabulary (not necessary, but gives faster training)
  local num_unk = 0
  local vecs = torch.Tensor(vocab.size, emb_dim)
  for i = 1, vocab.size do
    local w = string.gsub(vocab:token(i), '\\', '') -- remove escape characters
    if emb_vocab:contains(w) then
      vecs[i] = emb_vecs[emb_vocab:index(w)]
    else
      num_unk = num_unk + 1
      vecs[i]:uniform(-0.05, 0.05)
    end
  end
  print('unk count = ' .. num_unk)
  emb_vocab = nil
  emb_vecs = nil
  collectgarbage()

  -- load datasets
  print('Loading datasets')
  local train_dir = data_dir
  self.train_dataset = imagelstm.read_caption_dataset(train_dir, vocab, params.gpu_mode)

  collectgarbage()
  printf('Num train = %d\n', self.train_dataset.size)

  -- initialize model
  self.model = imagelstm.ImageCaptioner{
    batch_size = 33,
    emb_vecs = vecs,
    num_classes = vocab.size + 3, --For start, end and unk tokens
    gpu_mode = config.gpu_mode or false
  }
  self.batch_size = 33
end

function TrainChecks:check_train() 

  for i = 10, 11 do
    self:check_single_forward_pass(self.train_dataset, self.model)
    self:check_train_speed(i, self.train_dataset, self.model)
  end
end

function TrainChecks:check_single_forward_pass(dataset, model)
   --local idx = i + j - 1
   -- get the image features
   local start_time = sys.clock()
   local idx = 3
   local imgid = dataset.image_ids[idx]
   local image_feats = dataset.image_feats[imgid]

    -- get input and output sentences
   local sentence = dataset.sentences[idx]
   local out_sentence = dataset.pred_sentences[idx]

   local start1 = sys.clock()
   -- get text/image inputs
   local text_inputs = model.emb:forward(sentence)

   local start2 = sys.clock()
   local image_inputs = model.image_emb:forward(image_feats)

   -- get the lstm inputs
   local start3 = sys.clock()
   local inputs = model.lstm_emb:forward({text_inputs, image_inputs})
   
   -- compute the loss
   local start4 = sys.clock()
   inputs = torch.rand(10, 300)
   local lstm_output, class_predictions, caption_loss = model.image_captioner:forward(inputs, out_sentence)

   -- compute the input gradients with respect to the loss
   local start5 = sys.clock()
   local input_grads = model.image_captioner:backward(inputs, lstm_output, class_predictions, out_sentence)

   -- backprop the gradients through the linear combination step
   local start6 = sys.clock()
   local end_time = sys.clock()
   print("===== CHECKING SINGLE PASS ====")
   print("Time difference for single pass: ", start5 - start4)
   
end

-- Checks training with full dataset
function TrainChecks:check_train_speed(batch_size, dataset, model)
  local indices = torch.randperm(dataset.size)
  local zeros = torch.zeros(model.mem_dim)
  local tot_loss = 0
  local i = 3
    local feval = function(x)
      model.grad_params:zero()
      model.emb:zeroGradParameters()
      model.image_emb:zeroGradParameters()

      local start = sys.clock()
      local tot_forward_diff = 0
      local tot_backward_diff = 0
      local loss = 0
      for j = 1, batch_size do
        local idx = indices[i + j - 1]
        
        --local idx = i + j - 1
        -- get the image features
        local imgid = dataset.image_ids[idx]
        local image_feats = dataset.image_feats[imgid]

        -- get input and output sentences
        local sentence = dataset.sentences[idx]
        local out_sentence = dataset.pred_sentences[idx]

        local sentence = torch.ones(sentence:size())
        local out_sentence = torch.ones(out_sentence:size())

        if self.gpu_mode then
          --sentence = sentence:cuda()
          out_sentence = out_sentence:cuda()
          image_feats = image_feats:cuda()
        end

        local start1 = sys.clock()
        -- get text/image inputs
        local text_inputs = model.emb:forward(sentence)

        local start2 = sys.clock()
        local image_inputs = model.image_emb:forward(image_feats)

        -- get the lstm inputs
        local start3 = sys.clock()
        local inputs = model.lstm_emb:forward({text_inputs, image_inputs})

        -- compute the loss
        local start4 = sys.clock()
        local lstm_output, class_predictions, caption_loss = model.image_captioner:forward(inputs, out_sentence)
        loss = loss + caption_loss

        -- compute the input gradients with respect to the loss
        local start5 = sys.clock()
        local input_grads = model.image_captioner:backward(inputs, lstm_output, class_predictions, out_sentence)

        -- backprop the gradients through the linear combination step
        local start6 = sys.clock()

        tot_forward_diff = tot_forward_diff + start5 - start4
        tot_backward_diff = tot_backward_diff + start6 - start5
        local input_emb_grads = model.lstm_emb:backward({text_feats, image_feats}, input_grads)

        -- Separate gradients into word embedding and feature gradients
        local emb_grads = input_emb_grads[1]
        local image_grads = input_emb_grads[2]

        -- Do backward pass on image features and word embedding gradients
        local start7 = sys.clock()
        model.emb:backward(sentence, emb_grads)
        model.image_emb:backward(image_feats, image_grads)

        --  start5 - start4, start4 - start3, start3 - start2, start2 - start1)
      end
      local start8 = sys.clock()
      print("======BATCH SIZE ", batch_size, " ===============")
      print("Image captioner time ", tot_forward_diff / batch_size)
    end

end




