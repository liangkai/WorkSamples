require('torch')
require('xlua')
require('json')

local feature_path = arg[1]
local save_path = arg[2]

print('Converting ' .. feature_path .. ' to Torch serialized format')

print('Reading image features from ' .. feature_path)

local features = json.load(feature_path)
local feature_size = #features[1]
local num_features = #features
local vecs = torch.Tensor(num_features, feature_size)
for i = 1, num_features do
  for j = 1, feature_size do 
    vecs[i][j] = features[i][j]
  end
end
  
print('Done reading image features from ' .. feature_path)
torch.save(save_path, vecs)
print('Done saving image features')




