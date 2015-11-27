local CRowJoinTable, parent = torch.class('imagelstm.CRowJoinTable', 'nn.Module')

function CRowJoinTable:__init(dimension, nInputDims)
   parent.__init(self)
   self.size = torch.LongStorage()
   self.dimension = dimension
   self.gradInput = {}
   self.nInputDims = nInputDims
end

function CRowJoinTable:updateOutput(input) 
   local dimension = self.dimension
   if self.nInputDims and input[1]:dim()==(self.nInputDims+1) then
       dimension = dimension + 1
   end

   for i=1,#input do
      local currentOutput = input[i]
      local currSize
      if currentOutput:nDimension() == 1 then
         currSize = currentOutput:size(1)
      else
         currSize = currentOutput:size(dimension)
      end
      if i == 1 then
         self.size:resize(currentOutput:dim()):copy(currentOutput:size())
      else
         self.size[dimension] = self.size[dimension] + currSize
      end 
   end
   self.output:resize(self.size)
   
   local offset = 1  
   for i=1,#input do
      local currentOutput = input[i]
      local currSize
      if currentOutput:nDimension() == 1 then
         currSize = currentOutput:size(1)
      else
         currSize = currentOutput:size(dimension)
      end
      tmp = self.output:narrow(dimension, offset,
         currSize)

      if currentOutput:nDimension() == 1 then
        for j=1, tmp:size(1) do
          tmp[j]:copy(currentOutput)
        end
      else 
        tmp:copy(currentOutput)
      end
      offset = offset + currSize
   end
   return self.output
end

function CRowJoinTable:updateGradInput(input, gradOutput)
   local dimension = self.dimension
   if self.nInputDims and input[1]:dim()==(self.nInputDims+1) then
       dimension = dimension + 1
   end

   for i=1,#input do 
      if self.gradInput[i] == nil then
         self.gradInput[i] = input[i].new()
      end
      self.gradInput[i]:resizeAs(input[i])
   end

   local offset = 1
   for i=1,#input do
      local currSize
      local currentOutput = input[i]
      if currentOutput:nDimension() == 1 then
         currSize = currentOutput:size(1)
      else
         currSize = currentOutput:size(dimension)
      end
      tmp = gradOutput:narrow(dimension, offset,
         currSize)

      local currentGradInput = gradOutput:narrow(dimension, offset,
                      currSize)

      if currentOutput:nDimension() == 1 then
        local summedGrad = torch.squeeze(torch.sum(currentGradInput, 1))
        self.gradInput[i]:copy(summedGrad)
      else 
        self.gradInput[i]:copy(currentGradInput)
      end 
      offset = offset + currSize
   end
   return self.gradInput
end

function CRowJoinTable:type(type)
   self.gradInput = {}
   return parent.type(self, type)
end