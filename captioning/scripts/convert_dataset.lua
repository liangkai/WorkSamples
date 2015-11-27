require('torch')
require('xlua')
require('json')

local feature_path = arg[1]
local save_path = arg[2]

print('Converting ' .. feature_path .. ' to Torch serialized format')

print('Reading json dataset from ' .. feature_path)

local features = json.load(feature_path)
  
print('Done reading json dataset from ' .. feature_path)
torch.save(save_path, features)
print('Done saving json dataset')




