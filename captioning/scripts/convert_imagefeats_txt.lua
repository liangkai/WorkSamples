require('torch')
require('xlua')

local path = arg[1]
local vecpath = arg[2]
local prefix_toks = stringx.split(path, '.')
print('Converting ' .. path .. ' to Torch serialized format')

-- get dimension and number of lines
local file = io.open(path, 'r')
local line
local count = 0
local dim = 0
while true do
  line = file:read()
  if not line then break end
  if count == 0 then
    dim = #stringx.split(line)
  end
  count = count + 1
end

print('count = ' .. count)
print('dim = ' .. dim)

-- convert to torch-friendly format
file:seek('set')
local vecs = torch.DoubleTensor(count, dim)
for i = 1, count do
  xlua.progress(i, count)
  local tokens = stringx.split(file:read())
  for j = 1, dim do
    vecs[{i, j}] = tonumber(tokens[j])
  end
end
file:close()
torch.save(vecpath, vecs)