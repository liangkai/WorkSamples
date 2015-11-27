--[[

  Add a vector to every first row of a matrix.

  Input: { [n x m], [m] }

  Output: [n x m]

--]]

local CRowSingleTable, parent = torch.class('imagelstm.CRowSingleTable', 'nn.Module')

function CRowSingleTable:__init()
   parent.__init(self)
   self.gradInput = {}
end

function CRowSingleTable:updateOutput(input)
   self.output:resizeAs(input[1]):copy(input[1])
   self.output[1]:add(input[2])
   return self.output
end

function CRowSingleTable:updateGradInput(input, gradOutput)
   self.gradInput[1] = self.gradInput[1] or input[1].new()
   self.gradInput[2] = self.gradInput[2] or input[2].new()
   self.gradInput[1]:resizeAs(input[1])
   self.gradInput[2]:resizeAs(input[2]):zero()

   self.gradInput[1]:copy(gradOutput)
   self.gradInput[2]:add(gradOutput[1])

   return self.gradInput
end
