--[[

  ImageCaption-LSTM training script for image caption generation

--]]

require('.')

-- parse input params
params = {}
params.batch_size = 33
params.image_dim = 1024
params.emb_dim = 600
params.mem_dim = 600
params.learning_rate = 0.01
params.data_dir = 'data'
params.dataset = 'coco'
params.emb_dir = 'data/glove/'
params.combine_module = 'embedlayer'
params.hidden_module = 'hiddenlayer'
params.num_layers = 1
params.load_model = true
params.dropout = true
params.num_epochs = 100
params.epochs = 98
params.gpu_mode = false


local use_gpu_mode = params.gpu_mode or false
local num_epochs = params.epochs

add_unk = true
num_unk_sentences = 0
if use_gpu_mode then 
  print("Loading gpu modules")
  require('cutorch') -- uncomment for GPU mode
  require('cunn') -- uncomment for GPU mode
end

-- Calculates BLEU scores on train, test and val sets
function evaluate_results(test_model, beam_size, dataset)
    -- evaluate
  header('Evaluating on test set')
  test_save_path = 'output_test' .. os.clock() .. '.pred'
  evaluate(test_model, beam_size, test_dataset, 'predictions/bleu/' .. dataset .. '/' .. test_save_path)

  train_save_path = 'output_train' .. os.clock() .. '.pred'
  header('Evaluating on train set')
  evaluate(test_model, beam_size, train_dataset, 'predictions/bleu/' .. dataset .. '/' .. train_save_path)

  val_save_path = 'output_val' .. os.clock() .. '.pred'
  header('Evaluating on val set')
  evaluate(test_model, beam_size, val_dataset, 'predictions/bleu/' .. dataset .. '/' .. val_save_path)

  os.execute("./test.sh " ..  train_save_path .. ' ' .. val_save_path .. ' '
    .. test_save_path)

end

-- evaluates the model on the test set
function evaluate(model, beam_size, dataset, save_path)
  printf('-- using model with train score = %.4f\n', 0.00)
  --if model.gpu_mode then
  --   model:set_cpu_mode()
  --end

  local test_predictions = model:predict_dataset(dataset, beam_size, dataset.num_images)

  --if model.gpu_mode then
  --  model:set_gpu_mode()
  --end
  print("Saving predictions to ", save_path)
  model:save_predictions(save_path, loss, test_predictions)
end


function accuracy(pred, gold)
  return torch.eq(pred, gold):sum() / pred:size(1)
end

if params.dataset == 'coco' then
  print("Loading coco dataset")
  vocab, train_dataset, val_dataset, test_dataset =
    imagelstm.read_coco_dataset(params.data_dir .. "/coco/", params.gpu_mode)
else
  vocab, train_dataset, val_dataset, test_dataset = 
    imagelstm.read_flickr8k_dataset(params.data_dir .. "/flickr8k/", params.gpu_mode)
end


model = imagelstm.ImageCaptioner.load("model_coco_7.th")

if params.gpu_mode then 
  model:set_gpu_mode()
end

model:print_config()
--model:set_gpu_mode()
--train_predictions = model:get_sentences(model:predict_dataset(train_dataset, 5, 30))
--test_delpredictions = model:get_sentences(model:predict_dataset(test_dataset, 5, 30))
--val_predictions = model:get_sentences(model:predict_dataset(test_dataset, 5, 30))
evaluate_results(model, 5, 'coco')