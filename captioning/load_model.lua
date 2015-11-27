--[[

  ImageCaption-LSTM training script for image caption generation

--]]

require('.')

-- parse input params
params = {}
params.batch_size = 33
params.image_dim = 1024
params.emb_dim = 100
params.mem_dim = 100
params.learning_rate = 0.01
params.data_dir = 'data/'
params.emb_dir = 'data/glove/'
params.combine_module = 'embedlayer'
params.hidden_module = 'projlayer'
params.num_layers = 1
params.load_model = true
params.dropout = true
params.num_epochs = 100
params.epochs = 98
params.gpu_mode = false

num_unk_sentences = 0
num_tokens_per_sentence = 0
add_unk = true
vocab, train_dataset, val_dataset, test_dataset = 
    imagelstm.read_flickr8k_dataset(params.data_dir .. "/flickr8k/", params.gpu_mode)


model = imagelstm.ImageCaptioner{  batch_size = params.batch_size,
  optim_method = opt_method,
  vocab = vocab,
  hidden_dropout_prob = params.hidden_dropout_prob,
  in_dropout_prob = params.in_dropout_prob,
  dropout = params.dropout,
  num_layers = params.num_layers,
  num_classes = vocab.size,
  emb_dim = params.emb_dim,
  combine_module = params.combine_module,
  hidden_module = params.hidden_module,
  learning_rate = params.learning_rate,
  reg = params.reg,
  image_dim = params.image_dim,
  mem_dim = params.mem_dim,
  num_classes = vocab.size, --For start, end and unk tokens
  gpu_mode = use_gpu_mode -- Set to true for GPU mode
}

model:save("model_saved.th")
test_model = imagelstm.ImageCaptioner.load("model_saved.th")

-- load example sentences
ex_image = torch.rand(1024)
ex_in_sentence = torch.IntTensor{1, 5, 6, 7, 8}
ex_out_sentence = torch.IntTensor{5, 6, 7, 8, 2}

function forward_model(f_model, f_image, f_in_sentence, f_out_sentence)
        f_model:disable_dropouts()
        -- get text/image inputs
        local inputs = f_model.combine_layer:forward(f_in_sentence, f_image)
        local hidden_inputs = f_model.hidden_layer:forward(f_image)
        local lstm_output, class_predictions, caption_loss = 
        f_model.image_captioner:forward(inputs, hidden_inputs, f_out_sentence)
        return inputs, hidden_inputs, lstm_output, class_predictions, caption_loss
end

function norm_diff(a, b)
  return (a - b):norm()
end
inputs, hidden_inputs, lstm_out, class_pred, cap_loss = forward_model(model, ex_image, ex_in_sentence, ex_out_sentence)

inputs1, hidden_inputs1, lstm_out1, class_pred1, cap_loss1 = forward_model(test_model, ex_image, ex_in_sentence, ex_out_sentence)




