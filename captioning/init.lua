require('torch')
require('nn')
require('nngraph')
require('optim')
require('xlua')
require('sys')
require('lfs')
require('io')
require('json')

imagelstm = {}

-- Math helper functions
include('util/math.lua')

-- For reading word embeddings, image features, and captions
include('util/Vocab.lua')
include('util/read_data.lua') 

-- New forward/backward modules
include('modules/CRowAddTable.lua')
include('modules/CRowSingleTable.lua')
include('modules/CRowJoinTable.lua')

-- New input layers
include('input_layers/InputLayer.lua')
include('input_layers/AddLayer.lua')
include('input_layers/SingleAddLayer.lua')
include('input_layers/ConcatProjLayer.lua')
include('input_layers/ConcatLayer.lua')
include('input_layers/EmbedLayer.lua')
include('input_layers/GoogleEmbedLayer.lua')

-- New hidden layers
include('hidden_layers/HiddenLayer.lua')
include('hidden_layers/HiddenProjLayer.lua')
include('hidden_layers/HiddenDummyLayer.lua')
include('hidden_layers/HiddenSingleProjLayer.lua')

-- models
include('models/ImageCaptioner.lua')
include('models/ImageCaptionerLSTM.lua')
include('models/GoogleImageCaptioner.lua')
include('models/GoogleImageCaptionerLSTM.lua')
include('models/LSTM_Encoder.lua')
include('models/LSTM_Decoder.lua')

printf = utils.printf

-- global paths (modify if desired)
imagelstm.data_dir        = 'data'
imagelstm.models_dir      = 'trained_models'
imagelstm.predictions_dir = 'predictions'

-- share parameters of nngraph gModule instances
function share_params(cell, src, ...)
  for i = 1, #cell.forwardnodes do
    local node = cell.forwardnodes[i]
    if node.data.module then
    	--print(node.data.module)
      node.data.module:share(src.forwardnodes[i].data.module, ...)
    end
  end
end

function header(s)
  print(string.rep('-', 80))
  print(s)
  print(string.rep('-', 80))
end

print("Done loading modules for image captioner")

