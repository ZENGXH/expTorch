-----------------------------------------------------------------------
--[[ View ]]-- 
-- Abstract class
-- Adapter (design pattern) for torch.Tensor 
-- the principle is that the [view] and torch.Tensor data are binding together
-- in the view class
------------------------------------------------------------------------
local View, parent = torch.class("dp.View", "dp.Module")
View.isView = true

function View:__init(name)
   -- caches
   local name = name or 'view'
   self.log = dp.log() --loadfile(paths.concat(dp.DPRNN_DIR, 'utils', 'log.lua'))()
   self.log:trace('[init] view with name ', name)
   self.log:SetLoggerName(name)
   self._tensors = {}
   self._warn = false
end

local function getType(arg)
   local arg_type = torch.type(arg)
   if arg_type == 'table' then
      return getType(arg[1])
   end
   return arg_type
end

-- view is a string or a table of strings
function View:forward(view, inputORtype)
   view = view or 'default'
   local arg_type = getType(inputORtype)
   if arg_type == 'string' or arg_type == 'nil' then
      self.log:trace('{view} forward GET view: ', view, inputORtype)
      return self:forwardGet(view, inputORtype)
  end
   self.log:trace('{view} forward PUT view: ', view, ' with tensor: ', dp.helper.PrintSize(inputORtype))
   return self:forwardPut(view, inputORtype)
end


-- This method should be called by a maximum of one input Model.
-- It is assumed that any input Tensor to forward is represented as
-- the most expanded size of the orignal data. For example, an image
-- batch would be forwarded with its 4 dimensions, and never with 
-- collapses dimensions (2D). 
function View:forwardPut(view, input)
   error"Not Implemented"
end
   
-- This method could be called from multiple output Models
function View:forwardGet(view, tensor_type)
   error"Not Implemented"
end

-- view is a string or a table of strings
function View:backward(view, gradOutputORtype)
   local arg_type = getType(gradOutputORtype)
   if arg_type == 'string' or arg_type == 'nil' then
      return self:backwardGet(view, gradOutputORtype)
   end
   return self:backwardPut(view, gradOutputORtype)
end

-- This method could be called from multiple output Models
function View:backwardPut(view, gradOutput)
   error"Not Implemented"
end

-- This method should be called by a maximum of one input Model.
-- In the case of multiple output models having called backwardPut, 
-- the different gradInputs must be accumulated (sum grads).
function View:backwardGet(view, tensor_type)
   error"Not Implemented"
end

-- This method return the index of axis_char in string view
function View:FindAxis(axis_char, view)
   view = view or self._view
   local axis_pos = view:find(axis_char)
   if not axis_pos then
      error("Provided view '"..view.."' has no axis '"..axis_char.."'", 2)
   end
   return axis_pos
end

function View:findAxis(axis_char, view)
    return self:FindAxis(axis_char, view)
end

function View:IsBatchFirst()
    return self:FindAxis('b') == 1
end

-- Returns number of samples
function View:nSample()
   error"Not Implemented"
end

-- Returns a subet of data. 
-- If v is provided, copy indexed elements into its existing memory.
-- However, use the metadata found in self.
-- Providing v is faster (no memory allocation).
function View:index(v, indices)
   error"Not Implemented"
end

function View:sub(v, start, stop)
   error"Not Implemented"
end

-- return iterator over component Views
function View:pairs()
   error"Not Implemented"
end

-- return current view of data
function View:default()
   error"Not Implemented"
end

-- Changes the internal type of the data.
-- different behavior than torch.Tensor:type() (torch returns a tensor
-- with new type and keeps old)
function View:type(type)
   error"Not Implemented"
end

function View:float()
   return self:type('torch.FloatTensor')
end

function View:double()
   return self:type('torch.DoubleTensor')
end

function View:cuda()
   return self:type('torch.CudaTensor')
end

function View:toModule()
   error"Not Implemented"
end

function View:view()
   return self._view or ''
end
function View:GetDefaultViewStr()
   assert(self._view)
   return self._view
end

function View:SetViewStr(view)
    assert(torch.type(view) == 'string')
    self._view = view
    return self._view
end
function View:flush()
   error"Not Implemented"
end

---- static methods ----
--returns true if all indices in obj_table are instances of View
--else return false and index of first non-element
function View.areInstances(obj_table)
   local map = _.values(
      _.map(obj_table, 
         function(key, obj)
            return torch.isTypeOf(obj, 'dp.View')
         end
      )
   )
   return _.all(map, function(k,v) return v; end), _.indexOf(map, false)
end

function View.assertInstances(obj_table)
   local areInstances, index = View.areInstances(obj_table)
   index = index or 0
   assert(areInstances, "Error : object at index " .. index .. 
      " is of wrong type. Expecting type dp.View.")
end
