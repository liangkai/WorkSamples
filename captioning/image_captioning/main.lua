--[[

  ImageCaption-LSTM training script for image caption generation

--]]

require('..')

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Image Captioning Parameters')
cmd:text('Options')
cmd:option('-gpu_mode', false, 'gpu mode')
cmd:option('-optim', 'rmsprop', 'optimization type')
cmd:option('-epochs', 10,'number of epochs')
cmd:option('-dropout', false, 'use dropout')
cmd:option('-load_model', false, 'load model')
cmd:option('-batch_size', 33, 'batch_size')
cmd:option('-beam_size', 6, 'beam size for predictions')
cmd:option('-image_dim', 1024, 'input image size into captioner')
cmd:option('-emb_dim', 100, 'embedding size')
cmd:option('-mem_dim', 150,'memory dimension of captioner')
cmd:option('-learning_rate', 0.01, 'learning rate')
cmd:option('-data_dir', 'data', 'directory of caption dataset')
cmd:option('-dataset', 'coco', 'what dataset to use [flickr8k][coco]')
cmd:option('-emb_dir', 'data/flickr8k/', 'director of word embeddings')
cmd:option('-combine_module', 'addlayer', '[embedlayer] [addlayer] [singleaddlayer] [concatlayer] [concatprojlayer]')
cmd:option('-hidden_module', 'hiddenlayer', '[hiddenlayer]')
cmd:option('-model_epoch', 98, 'epoch to load model from')
cmd:option('-num_layers', 4, 'number of layers in lstm network')
cmd:option('-in_dropout_prob', 0.5, 'probability of input dropout')
cmd:option('-hidden_dropout_prob', 0.5, 'probability of hidden dropout')
cmd:text()

-- parse input params
params = cmd:parse(arg)

local use_gpu_mode = params.gpu_mode or false
local num_epochs = params.epochs

if use_gpu_mode then 
  print("Loading gpu modules")
  require('cutorch') -- uncomment for GPU mode
  require('cunn') -- uncomment for GPU mode
end

function accuracy(pred, gold)
  return torch.eq(pred, gold):sum() / pred:size(1)
end

local opt_method
if params.optim == 'adagrad' then
  opt_method = optim.adagrad
elseif params.optim == 'rmsprop' then
  opt_method = optim.rmsprop
else
  opt_method = optim.sgd
end

header('Image-Captioning with LSTMs')

local vocab, train_dataset, val_dataset, test_dataset 

if params.dataset == 'coco' then
  print("Loading coco dataset")
  vocab, train_dataset, val_dataset, test_dataset =
    imagelstm.read_coco_dataset(params.data_dir .. "/coco/", params.gpu_mode)
else
  vocab, train_dataset, val_dataset, test_dataset = 
    imagelstm.read_flickr8k_dataset(params.data_dir .. "/flickr8k/", params.gpu_mode)
end

-- load embeddings
print('Loading word embeddings')
local emb_dir = params.emb_dir
local emb_prefix = emb_dir .. 'vocab_feats'
local emb_vocab, emb_vecs = imagelstm.read_embedding(emb_prefix .. '.vocab', emb_prefix .. '.600d.th')
local emb_dim = emb_vecs:size(2)

-- use only vectors in vocabulary (not necessary, but gives faster training)
local num_unk = 0
local vecs = torch.ones(vocab.size, params.emb_dim)
for i = 1, vocab.size do
  local w = string.gsub(vocab:token(i), '\\', '') -- remove escape characters
  if emb_vocab:contains(w) then
     local mean_vec = emb_vecs[emb_vocab:index(w)][1] -- want to make sure not all emb same
     vecs[i] = vecs[i] * mean_vec
     --vecs[i]:uniform(-0.05 + mean_vec, 0.05 + mean_vec)
  else
    num_unk = num_unk + 1
    vecs[i]:uniform(-0.05, 0.05)
  end
end

print('unk count = ' .. num_unk)
emb_vocab = nil
emb_vecs = nil

collectgarbage()
printf('num train = %d\n', train_dataset.size)

-- initialize model
local model = imagelstm.ImageCaptioner{
  batch_size = params.batch_size,
  optim_method = opt_method,
  emb_vecs = vecs,
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

local loss = 0.0

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
  printf('-- using model with train score = %.4f\n', loss)
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


-- print information
header('model configuration')
printf('max epochs = %d\n', num_epochs)
model:print_config()

local model_save_path = string.format(
  imagelstm.models_dir .. model:getPath(num_epochs))

-- local test_model = imagelstm.GoogleImageCaptioner.load('model.th')
--for j = 1, 20 do 
--print("======== BEAM SIZE " .. j .. " ===========")
-- evaluate_results(test_model, 1, 'flickr8k')
--end

if params.load_model then
--if true then
  print("Loading model from file " .. model_save_path)
  model = imagelstm.ImageCaptioner.load(model_save_path) -- uncomment to load model
end

if lfs.attributes(imagelstm.models_dir) == nil then
  lfs.mkdir(imagelstm.models_dir)
end

if lfs.attributes(imagelstm.predictions_dir) == nil then
  lfs.mkdir(imagelstm.predictions_dir)
end
-- train
local train_start = sys.clock()
local best_train_score = -1.0
local best_train_model = model

    -- save them to disk for later use
local predictions_save_path = string.format(
imagelstm.predictions_dir .. model:getPath(2))

function test_saving()
  local predictions_save_path = string.format(
  imagelstm.predictions_dir .. model:getPath(1))


  local test_predictions = model:predict_dataset(test_dataset, params.beam_size, 30)
  print("Saving predictions to ", predictions_save_path)
  model:save_predictions(predictions_save_path, loss, test_predictions)

  local model_save_path = string.format(
  imagelstm.models_dir .. model:getPath(i))

  local train_loss, perplexity = model:eval(train_dataset)
  printf("Train loss is %.4f Perplexity %.4f \n", train_loss, perplexity)

  print('Norm of items before saving ', model.params:norm())
  print('Norm of combine layers', model.combine_layer:getParameters():norm())
  print('Norm of hidden layers', model.hidden_layer:getParameters():norm())
  print('Norm of lstm layers', model.image_captioner:getParameters():norm())

  model:save(model_save_path)

  local train_loss, perplexity = model:eval(train_dataset)
  printf("Train loss is %.4f Perplexity %.4f \n", train_loss, perplexity)
  print('Norm of items right after saving ', model.params:norm())
  print('Norm of combine layers', model.combine_layer:getParameters():norm())
  print('Norm of hidden layers', model.hidden_layer:getParameters():norm())
  print('Norm of lstm layers', model.image_captioner:getParameters():norm())

  local new_model = imagelstm.ImageCaptioner.load(model_save_path)
  local train_loss, perplexity = new_model:eval(train_dataset)
  printf("Train loss AFTER LOADING SAVING is %.4f Perplexity %.4f \n", train_loss, perplexity)

  print('Norm of items before saving ', model.params:norm())
  print('Norm of combine layers', model.combine_layer:getParameters():norm())
  print('Norm of hidden layers', model.hidden_layer:getParameters():norm())
  print('Norm of lstm layers', model.image_captioner:getParameters():norm())
  assert(false)
end

header('Training Image Captioning LSTM')
for i = 1, params.epochs do

  local curr_epoch = i
  local start = sys.clock()
  printf('-- epoch %d\n', i)
  loss = model:train(train_dataset)

  printf("Average loss %.4f \n", loss)
  printf('-- finished epoch in %.2fs\n', sys.clock() - start)

  local train_loss, perplexity = model:eval(train_dataset)
  printf("Train loss is %.4f Perplexity %.4f \n", train_loss, perplexity)

  local test_loss, perplexity = model:eval(test_dataset)
  printf("Test loss is %.4f Perplexity %4.f \n", test_loss, perplexity)

  local val_loss, perplexity = model:eval(val_dataset)
  printf("Val loss is %.4f Perplexity %4.f \n", val_loss, perplexity)

  local model_save_path = string.format(
  imagelstm.models_dir .. model:getPath(i))
    -- save them to disk for later use


  print('writing model to ' .. model_save_path)
  model:save(model_save_path, curr_epoch)

  local predictions_save_path = string.format(
  imagelstm.predictions_dir .. model:getPath(i))


  local test_predictions = model:predict_dataset(test_dataset, params.beam_size, 30)
  print("Saving predictions to ", predictions_save_path)
  model:save_predictions(predictions_save_path, loss, test_predictions)


  if curr_epoch % 20 == 10 then
    print('writing model to ' .. model_save_path)
    evaluate_results(model, 1, params.dataset)
  end
  --model = imagelstm.ImageCaptioner.load(model_save_path)

end

-- write model to disk
  
local model_save_path = string.format(
  imagelstm.models_dir .. model:getPath(params.epochs))
print('writing model to ' .. model_save_path)
model:save(model_save_path)

-- to load a saved model
-- local loaded = imagelstm.ImageCaptioner.load(model_save_path)
