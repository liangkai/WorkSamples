--[[

  Math helper functions on tables

--]]

-- Check type of input
function check_type(input, desired_type)
  local input_type = torch.typename(input)
  assert(input_type == desired_type, "input has type " .. input_type ..
   " but desired is " .. desired_type)
end

-- Enable dropouts
function enable_sequential_dropouts(model)
   for i,m in ipairs(model.modules) do
      if m.module_name == "nn.Dropout" or torch.typename(m) == "nn.Dropout" then
        m:training()
      end
   end
end

-- Disable dropouts
function disable_sequential_dropouts(model)
   for i,m in ipairs(model.modules) do
      if m.module_name == "nn.Dropout" or torch.typename(m) == "nn.Dropout" then
        m:evaluate()
      end
   end
end

-- Convert 1-d torch tensor to lua table
function tensor_to_array(t1)
  -- This assumes `t1` is a 2-dimensional tensor!
  local t2 = {}
  for i=1,t1:size(1) do
    t2[i] = t1[i]
  end
  return t2
end


-- Sorts tables by first value
-- first_entry, second_entry are tables
function min_sort_function(first_table, second_table)
    return first_table[1] < second_table[1]
end


-- Sorts tables by first value
-- first_entry, second_entry are tables
function max_sort_function(first_table, second_table)
    return first_table[1] > second_table[1]
end

-- Argmax: hacky way to ignore end token to reduce silly sentences
function argmax(v, ignore_end)
  local vals, indices = torch.max(v, 1)
  return indices[1]
end

-- TopkArgmax returns top k indices, values from list
function topk(list, k)
  tmp_list = {}
  for i = 1, #list do
    table.insert(tmp_list, list[i])
  end
  table.sort(tmp_list, max_sort_function)

  max_entries = {}
  for i = 1, k do
    table.insert(max_entries, tmp_list[i])
  end

  return max_entries
end

-- TopkArgmax returns top k indices, values from list
function topkargmax(list, k)
  local cloned_list = list:clone()
  max_indices = {}
  for i = 1, k do
    local vals, indices = torch.max(cloned_list, 1)
    local best_index = indices[1]
    cloned_list[best_index] = -1000
    table.insert(max_indices, best_index)
  end
  return max_indices
end
