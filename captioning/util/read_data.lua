--[[

  Functions for loading data from disk.

--]]
  
function imagelstm.read_embedding(vocab_path, emb_path)
  -- Reads vocabulary from vocab_path, 
  -- Reads word embeddings from embd_path
  local vocab = imagelstm.Vocab(vocab_path, false)
  local embedding = torch.load(emb_path)
  return vocab, embedding
end

function imagelstm.read_sentences(path, vocab)
  -- Reads sentences from specified path
  -- Reads vocab from vocab path
  local sentences = {}
  local file = io.open(path, 'r')
  local line
  while true do
    line = file:read()
    if line == nil then break end
    local tokens = stringx.split(line)
    local len = #tokens
    local sent = torch.IntTensor(len)
    for i = 1, len do
      local token = tokens[i]
      sent[i] = vocab:index(token)
    end
    sentences[#sentences + 1] = sent
  end

  file:close()
  return sentences
end


--[[

  Functions for loading caption data from disk.

--]]
function imagelstm.read_dataset(dataset_path)
  -- Reads image features from disk
  -- Returns vocab, embeddings

  print('Reading caption dataset from ' .. dataset_path)
  --local dataset = json.load(dataset_path)
  local dataset = torch.load(dataset_path)
  local images = dataset['images']

  -- TODO: preprocess images
  print('Done reading caption dataset from ' .. dataset_path)
  return images
end

function imagelstm.read_image_features(feature_path)
  -- Reads all image features from file
  -- Returns Torch tensor of size (image_feature_size)

  print('Reading image features from ' .. feature_path)
  vecs = torch.load(feature_path)
  print('Done reading image features from ' .. feature_path)

  return vecs
end

--[[

 Read captions dataset

--]]

-- reads caption sentences from directory
-- dir: where caption sentences are
-- desired_split: whether to read train, test sentences
function imagelstm.read_caption_sentences(dir, desired_split)
  local caption_dataset = {}

  local annotation_dataset = imagelstm.read_dataset(dir .. 'dataset.th')
  local num_images = #annotation_dataset
  local num_desired_images = 0
  -- get input and output sentences
  local sentences = {}

  -- min number of references
  local min_references = 10
  for i = 1, num_images do
    curr_image = annotation_dataset[i]
    local split = curr_image['split']
    if split == desired_split then 
      num_desired_images = num_desired_images + 1
      local curr_sentences = {}
      local curr_sentences = curr_image['sentences']
      for j = 1, #curr_sentences do
          min_references = math.min(#curr_sentences, min_references)
          local tokens = curr_sentences[j]['tokens']
          table.insert(curr_sentences, tokens)
      end
      table.insert(sentences, curr_sentences)
    end
  end
  
  caption_dataset.sentences = sentences
  caption_dataset.num_images = num_desired_images
  caption_dataset.size = num_desired_images
  return caption_dataset, min_references
end

-- Reads caption dataset from specified file
-- Directory is where caption dataset resides
-- Vocabulary maps tokens to ids
-- Gpu_mode determines whether to load the tensors as float or Cuda
-- Desired split: Whether to load the train or validation
function imagelstm.read_caption_dataset(dir, vocab, gpu_mode, desired_split)
  local caption_dataset = {}

  local annotation_dataset = imagelstm.read_dataset(dir .. 'dataset.th')
  local num_images = #annotation_dataset
  local num_desired_images = 0 -- Only include images that are in the split
  -- get input and output sentences
  local sentences = {}
  local out_sentences = {}
  local image_ids = {}
  local max_num_sent_tokens = 0
  local max_num_tokens = 0
  -- single image ids with no duplicates
  local single_image_ids = {}
  for i = 1, num_images do
    local curr_image = annotation_dataset[i]
    local split = curr_image['split']
    -- dataset is zero indexed, torch is 1 indexed
    local curr_imgid = curr_image['imgid'] + 1
    local curr_sentences = curr_image['sentences']

    if split == desired_split then
      num_desired_images = num_desired_images + 1
      table.insert(single_image_ids, curr_imgid)
      for j = 1, #curr_sentences do
        local split = curr_sentences[j]['split']
        local tokens = curr_sentences[j]['tokens']
        if #tokens > max_num_sent_tokens then
          max_num_sent_tokens = #tokens
        end
        if #tokens < 100 then
          table.insert(tokens, "</s>")

          -- first get labels for sentence
          local out_ids = vocab:map_no_unk(tokens)
          local out_sentence = table.concat(vocab:tokens(tensor_to_array(out_ids)), ' ')

          -- then get input sentence stuff: off by one language model
          table.insert(tokens, 1, "<s>")
          table.remove(tokens)
          local in_ids = vocab:map_no_unk(tokens)
          max_num_tokens = max_num_tokens + in_ids:size(1)

          local in_sentence = table.concat(vocab:tokens(tensor_to_array(in_ids)), ' ')

          --print("Out sentence " .. out_sentence)
          --print("In sentence " .. in_sentence)
          if gpu_mode then
            out_ids = out_ids:cuda()
            in_ids = in_ids:cuda()
          end

          -- then make a new one with special start symbol
          table.insert(image_ids, curr_imgid)
          table.insert(out_sentences, out_ids)
          table.insert(sentences, in_ids)
        end
        
      end
    end
  end

  print("Number of sentences to train on", #image_ids)
  print("Max sentence length", max_num_tokens, max_num_sent_tokens)
  image_feats = imagelstm.read_image_features(dir .. 'googlenet_feats.th')
  if gpu_mode then
    image_feats = image_feats:cuda()
  end

  caption_dataset.vocab = vocab
  caption_dataset.image_feats = image_feats
  caption_dataset.image_ids = image_ids
  caption_dataset.single_image_ids = single_image_ids
  caption_dataset.pred_sentences = out_sentences
  caption_dataset.sentences = sentences
  caption_dataset.size = #sentences
  caption_dataset.num_images = num_desired_images

  return caption_dataset
end

-- reads flickr8k dataset where base directory path is specified
-- base_path: base path of flickr8k
-- gpu_mode: whether to read in image features in gpu or cpu mode
-- returns: imagelstm.Vocab, train, val, test datasets in that order
function imagelstm.read_flickr8k_dataset(base_path, gpu_mode)
    -- directory containing dataset files
  local data_dir = base_path

  -- load vocab
  local vocab = imagelstm.Vocab(data_dir .. 'vocab.txt', true)

  -- load datasets

  -- load train dataset
  local train_dir = data_dir
  local train_dataset = imagelstm.read_caption_dataset(train_dir, vocab, gpu_mode,
    'train')

  -- load val dataset
  local val_dir = data_dir
  local val_dataset = imagelstm.read_caption_dataset(val_dir, vocab, gpu_mode, 'val')

  -- load test dataset
  local test_dir = data_dir
  local test_dataset = imagelstm.read_caption_dataset(test_dir, vocab, gpu_mode, 
    'test')

  return vocab, train_dataset, val_dataset, test_dataset
end

-- reads coco dataset where base directory path is specified
-- base_path: base path of flickr8k
-- gpu_mode: whether to read in image features in gpu or cpu mode
-- returns: imagelstm.Vocab, train, val, test datasets in that order
function imagelstm.read_coco_dataset(base_path, gpu_mode)
    -- directory containing dataset files
  local data_dir = base_path

  -- load vocab
  local vocab = imagelstm.Vocab(data_dir .. 'vocab.txt', true)

  -- load datasets

  -- load train dataset
  local train_dir = data_dir .. 'train/'
  local train_dataset = imagelstm.read_caption_dataset(train_dir, vocab, gpu_mode,
    'train')

  -- load test dataset
  local test_dir = data_dir .. 'test/'
  local test_dataset = imagelstm.read_caption_dataset(test_dir, vocab, gpu_mode, 
    'train')

    -- load val dataset
  local val_dir = data_dir .. 'val'
  --local val_dataset = imagelstm.read_caption_dataset(val_dir, vocab, params.gpu_mode, 'val')
  local val_dataset = test_dataset
  return vocab, train_dataset, val_dataset, test_dataset
end

