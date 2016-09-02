-----------------------------------------------------------------------
--[[ DataView ]]-- 
-- Allows for efficiently communicating tensors between Models.
-- Exists at the output of a Model or a DataSource
-- provide b() and bf() view, others can be found in the sub-class
------------------------------------------------------------------------
local DataView, parent = torch.class("dp.DataView", "dp.View")
DataView.isDataView = true

function DataView:__init(view, input, name)
   local name = name or 'DataView'
   parent.__init(self, name)
   self.log.trace(name, ' view init with view ', view)
   self._is_data_filled = false
   assert(view and input)
   -- require when init
   if view and input then
      self.log.trace('\t calling forward')
      self:forward(view, input)
   end
   self._module_graph = {}
end

---------------------- FORWARD -----------------------
-- This method should be called by a maximum of one input Model.
-- It is assumed that any input Tensor to forward is represented as
-- the most expanded size of the orignal data. For example, an image
-- batch would be forwarded with its 4 dimensions, and never with 
-- collapses dimensions (2D). 
--
-- @para view string describe the input(tensor) dimension information
-- @para input
-- fill self._tensors table(inherit from dp.View)
-- self._tensors = 
--      key: view(string) & value: viewTable,
--          viewTable: key: type(string) & value: torch.Tensor
--  {['bchw'] = {['torch.DoubleTensor'] = input}}
-- self._modules = 
--      key: view(string) & value: moduleTable
--          moduleTable = {modula, typeConversionTable}
--  {['bchw'] = { nn.Identity(),{['torch.DoubleTensor'] = nn.Identity()}}}
-----------------------------------------------------------------------
function DataView:forwardPut(view, input)
   self.log.trace('[DataView] fw PUT view: ', view, 
      ' with tensor size: ', dp.helper.PrintSize(input))
   -- store input for later use
   self._dim = #view -- eg #'bhwc' = 4
   dp.helper.Asserteq(input:dim(), self._dim, "view has more axes than input has dims, get view: "..view)
   
   if self._view and (view ~= self._view) then
      self._modules = nil
   end
   self:SetViewStr(view)
   self:SetInputTensor(input)
   -- since this method is called only once at beginning of batch,
   -- we reinitialize gradOutputs and tensors cache:
   self._type = torch.typename(input)
   self._tensors = {[view] = {[self._type] = input}}
   self._gradOutputs = {}
   if not self._modules then
      self._modules = {
         [view] = {nn.Identity(), {[self._type] = nn.Identity()}}
      }
      -- self_modules = {['bcwh']: {nn.Identity(), {'CudaTensor'=nn.Identity()}}
   end
end
   
-- This method could be called from multiple output Models
-- return the tensor from the self._tensor by key'view' and key'tensor_type'
function DataView:forwardGet(view, tensor_type)
   self._got = true
   tensor_type = tensor_type or self._type
   -- retrieve a viewTable
   local viewTable = self._tensors[view]
   if not viewTable then
      -- no such view in self._tensors table
      -- no viewTable: get tensor from module
      return self:tensorFromModule(view, tensor_type)
   end
   local tensor = viewTable[tensor_type]
   if not tensor then
      -- no such tensor_type in the view of self._tensors table
      return self:tensorFromModule(view, tensor_type)
   end
   return tensor
end

-- returns a tensor of shape view and type tensor_type
-- return the module from the view and the 
function DataView:tensorFromModule(view, tensor_type)
   local viewTable = self._tensors[view] or {}
   local input_type = torch.typename(self._input)
   local moduleTable = self._modules[view]

   if not moduleTable then
      -- no moduleTable: build a module
      local modula = self[view](self)
      -- make sure it accepts the right input type
      modula:type(self._type)
      local copy = nn.Copy(input_type, tensor_type)
      self._modules[view] = {modula, {[tensor_type] = copy}}
      local tensor = modula:forward(self._input)
      viewTable[input_type] = tensor
      tensor = copy:forward(tensor)
      viewTable[tensor_type] = tensor
      self._tensors[view] = viewTable
      return tensor
   end

   local modula, copierTable = unpack(moduleTable)
   local tensor = modula:forward(self._input)
   viewTable[input_type] = tensor
   local copier = copierTable[tensor_type]

   if not copier then
      -- no copier : build copier module
      copier = nn.Copy(input_type, tensor_type)
      copierTable[tensor_type] = copier
   end
   
   tensor = copier:forward(tensor)
   viewTable[tensor_type] = tensor
   self._tensors[view] = viewTable
   return tensor
end

------------------------ BACKWARD -------------------------

-- This method could be called from multiple output Models
function DataView:backwardPut(view, gradOutput)
   -- store gradOutput in list
   table.insert(self._gradOutputs, {view, gradOutput})
end

-- This method should be called by a maximum of one input Model.
-- In the case of multiple output models having called backwardPut, 
-- the different gradInputs must be accumulated (sum grads).
function DataView:backwardGet(view, tensor_type)
   if view and view ~= self._view then
      error(torch.type(self)..
         ": backwardGet should be called with same view used for "..
         "last forward (or nil) i.e. ".. (self._view or 'nil') .. 
         " not " .. (view or 'nil'))
   end
   if tensor_type and self._type ~= tensor_type then
      error(torch.type(self)..
         ": backwardGet sould be called with the same type as "..
         "forwarded input")
   end
   tensor_type = tensor_type or self._type
   
   local view, gradOutput, gradInput
   
   -- optimization : one-to-one backward
   if #self._gradOutputs == 1 then
      view, gradOutput = unpack(self._gradOutputs[1])
      tensor_type = torch.typename(gradOutput)
      local moduleTable = self._modules[view]
      assert(moduleTable, "backward must follow a forward")
      local modula, copyTable = unpack(moduleTable)
      assert(copyTable, "backward must follow a forward")
      local copy = copyTable[tensor_type]
      assert(copy, "backwardPut should have been called with same "..
         "type as its commensurate forwardGet or the forwardPut.")
      gradInput = copy:backward(self._input, gradOutput)
      gradInput = modula:backward(self._input, gradInput)
      return gradInput
   end
   
   -- slower : many-to-one backward
   if not self._gradOutputs or #self._gradOutputs == 0 then 
      error("Cannot backwardGet without a previous backwardPut", 2)
   end
   if not self._gradInput then
      self._gradInput = self._input:clone()
   end
   for i, gradOutputTable in ipairs(self._gradOutputs) do
      view, gradOutput = unpack(gradOutputTable)
      local moduleTable = self._modules[view]
      assert(moduleTable, "backward must follow a forward")
      local modula, copyTable = unpack(moduleTable)
      assert(copyTable, "backward must follow a forward")
      tensor_type = torch.typename(gradOutput)
      gradInput = copyTable[tensor_type]:backward(self._input, gradOutput)
      gradInput = modula:backward(self._input, gradInput)
      -- accumulate
      if i == 1 then
         self._gradInput:copy(gradInput)
      else
         self._gradInput:add(gradInput)
      end
   end
   return self._gradInput
end

---------------------- VIEWS ---------------------------

-- batch feature
function DataView:bf()
   local view, dim = self._view, self._dim
   local b_pos = self:findAxis('b', view)

   -- was b originally
   if dim == 1 then
      if self._warn then
         print("DataView:feature Warning: provided data has one "..
               "dim. Assuming dim is 'b' axis with feature size of 1")
      end
      return nn.Reshape(1)
   end

   -- was b...
   local modula
   -- make batchSize as the first dimention
   if b_pos ~= 1 then
      modula = nn.Transpose({1, b_pos})
   end
   if dim > 2 then
      local transpose = modula
      local reshape = nn.Reshape(self:sampleSize(b_pos))
      if transpose then -- b_pos ~= 1, need transpose
         modula = nn.Sequential()
         modula:add(transpose) -- make batchSize in first dimen
         modula:add(reshape) -- reshape all feature in one line
      else
         modula = reshape -- batchSize in first dimen already
      end
   end
   return modula or nn.Identity()
end

-- vector view. 
-- Only works with 'bf' with size(f) is 1 or 'b'
function DataView:b()
   local view, dim = self._view, self._dim
   local b_pos = self:findAxis('b', view)
   -- was bf
   if view == 'bf' then
      if self._input:size(2) ~= 1 then
         error("Cannot convert view bf with size(f) > 1 to b", 2)
      end
      return nn.Select(2, 1)
   elseif view ~= 'b' then
      error("Cannot convert view "..view.." to b", 2)
   end
   return nn.Identity()
end

-- returns the current view of the data
function DataView:default()
   return nn.Identity()
end

---------------- Module:toModule -----------------------

-- accumulates all forward modules into a table
function DataView:modulePut(fwd_module, view, tensor_type)
   self._put = true
   local viewTable = self._module_graph[view]
   if not viewTable then
      -- create a viewTable for view, init a fwd_module for tensor_type
      -- put the viewTable in to self._module_graph
      viewTable = {[tensor_type] = {fwd_module}}
      self._module_graph[view] = viewTable
      return
   end
   local moduleList = viewTable[tensor_type]
   if not moduleList then
      -- create fwd_module for tensor_type
      viewTable[tensor_type] = {fwd_module}
      return
   end
   -- already have viewTable for [view] in self._module_graph
   -- already have a fwd_module for the [tensor_type] in the viewTable
   -- append the additional fwd_module to the list
   table.insert(moduleList, fwd_module)
end

-- composes all forward modules into a single composite Module :
function DataView:moduleGet(bwd_module)
   if not self._got then
      if self._put or (not self._modules) then
         error"Model:toModule() should be preceded by a call to Model:forward()"
      end
      -- assume self is the output layer's output View
      return bwd_module
   end
   -- how many output Models use this?
   local nOut = 0
   local view, tensor_type, fwd_module
   for view_, viewTable in pairs(self._module_graph) do
      view = view_
      for tensorType, moduleList in pairs(viewTable) do
         nOut = nOut + #moduleList
         tensor_type = tensorType
         fwd_module = moduleList[1]
      end
   end
   local mlp = nn.Sequential()
   if bwd_module then -- the input View of the network has no bwd_module
      -- the backward (previous) module comes first
      mlp:add(bwd_module) 
   end
   if nOut == 1 then
      -- only 1 output Model: simple build
      local moduleTable = self._modules[view]
      local viewModule = moduleTable[1]
      if torch.type(viewModule) ~= 'nn.Identity' then
         mlp:add(viewModule)
      end
      if tensor_type ~= self._type then
         local typeModule = moduleTable[2][tensor_type]
         mlp:add(typeModule)
      end
      mlp:add(fwd_module)
      return mlp
   end
   
   print"DataView:moduleGet warning: untested multi-output code follows"
   -- else: multiple outputs : complicated build (output is a table)
   -- nn.Sequential(
   --    bwd_module,
   --    nn.ConcatTable( --concatView
   --       nn.Sequential(
   --          viewTable1, 
   --          nn.ConcatTable( --concatType
   --             nn.Sequential(
   --                typeTable1,
   --                nn.ConcatTable( --concatFwd
   --                   fwd_module1,
   --                   ... multi fwd
   --                )
   --             )
   --             ... multi type
   --          )
   --       )
   --       ... multi view
   --    ),
   --    nn.FlattenTable()
   -- )
   -- TODO : prune simpler graphs (single view, type or module branches)
   local concatView = nn.ConcatTable()
   for view, viewTable in pairs(self._module_graph) do
      local moduleTable = self._modules[view]
      local viewModule = moduleTable[1]
      local concatType = nn.ConcatTable()
      local seqView = nn.Sequential()
      seqView:add(viewModule)
      for tensor_type, moduleList in pairs(viewTable) do
         local typeModule = moduleTable[2][tensor_type]
         local seqType = nn.Sequential()
         seqType:add(typeModule)
         local concatFwd = nn.ConcatTable()
         for i, fwd_module in ipairs(moduleList) do
            concatFwd:add(fwd_module)
         end
         seqType:add(concatFwd)
         concatType:add(seqType)
      end
      seqView:add(concatType)
      concatView:add(seqView)
   end
   mlp:add(concatView)
   mlp:add(nn.FlattenTable())
end


---------------------- MISC ----------------------------

-- number of features in each sample
-- multiply all size of self._input except batchSize
function DataView:sampleSize(b_pos, view, data)
   b_pos = b_pos or self:findAxis('b', view)
   data = data or self._input
   local size = 1
   for i=1, data:dim() do
      if i ~= b_pos then
         size = size * self._input:size(i)
      end
   end
   return size
end

-- Returns number of samples
function DataView:nSample(b_pos)
   b_pos = b_pos or self:findAxis('b')
   return self._input:size(b_pos)
end

function DataView:pairs()
   return pairs{self}
end

function DataView:clone()
   return torch.protoClone(self, self._view, self._input:clone())
end

-- Used by dp.Preprocess instances to replace the input
-- see dp.ZCA:apply() for an example
function DataView:replace(view, output, inplace)
   self:backward(view, output)
   output = self:backward()
   if inplace then
      self._input:copy(output)
   else
      self:input(output)
   end
   self:flush()
   -- forwardPut it back in
   self:forward(self._view, self._input)
end

-- flush module and tensor cache
function DataView:flush()
   self._tensors = {}
   self._modules = nil
   self._module_graph = {}
   self._got = false
   self._put = false
end

----------------------------------------------------
--[[ filling view with data ]]--
-- use for LoadWholeDataSet
-- When v is provided, reuse its data (a torch.Tensor)
-----------------------------------------------------
function DataView:CreateSubViewWithIndex(indices)
   local data = self._input:index(b_pos, indices)
    v = torch.protoClone(self, self._view, data)
   return v
end

function DataView:FillSubViewWithIndex(v, indices)
   if torch.type(v) ~= torch.type(self) then
      error("Expecting "..torch.type(self).." at arg 1 "..
            "got "..torch.type(v).." instead")
   end  
   local data = v:input()
   data:index(self:input(), b_pos, indices)
   assert(self._view == v._view, "Expecting arg 1 to have same view")
   v:forward(self._view, data)
   return v
end

function DataView:index(v, indices)
   local b_pos = self:findAxis('b')
   local data
   if indices and v then
      return self:FillSubViewWithIndex(v, indices)
   end
   indices = indices or v
   return self:CreateSubViewWithIndex(indices)
end

-- Returns a sub-view narrowed on the batch dimension
-- inplace returns a narrow window into self._input instead of a copy
-- return a dataView which contain torch.Tensor data which is narrow from the original DataView

function DataView:FillSubViewWithSub(v, start, stop, inplace)
   if torch.type(v) ~= torch.type(self) then
      error("Expecting "..torch.type(self).." at arg 1 "..
      "got "..torch.type(v).." instead")
   end
   if v._view and self._view ~= v._view then
      error("Expecting arg 1 to have same view")
   end
   local b_pos = self:findAxis('b')
   local input = self._input:narrow(b_pos, start, stop-start + 1)
   local data = v:IsFilled() or self:input().new()
   if inplace then
      -- user beware: this doesn't interact well with :index()
      data:set(input) 
   else
      data:resizeAs(input)
      data:copy(input)
   end
   -- filling the sub slice of data into v and return
   v:forward(self._view, data)
   return v
end

function DataView:CreateSubViewWithSub(start, stop, inplace)
   local v = torch.protoClone(self)
   return self:FillSubViewWithSub(v, start, stop, inplace)
end

function DataView:sub(v, start, stop, inplace)
   local data
   if v and stop then
      -- given v and stop, check type v is the same with v; 
      -- check v._view == self._view
      return self:FillSubViewWithSub(v, start, stop, inplace)
   else -- no v, i.e receive 3 or 2 arguments only
      if v then -- the first arguments is indeed `start`
         -- stop not given, only receive two arguments
         inplace = stop
         stop = start
         start = v
      end
      -- create an object of DataView
      return self:CreateSubViewWithIndex(start, stop, inplace)
   end
end

-- optional : do sub inplace (no mem-copy), reuse returned dataview
function DataView:ipairsSub(batchSize, inplace, reuse)
   local nSample = self:nSample()
   local start = 1
   local nSampled = 0
   local stop
   local dv = reuse and torch.protoClone(self) or false
   -- build iterator
   return function()
      if nSampled >= nSample then
         return
      end
      dv = dv or torch.protoClone(self)
      stop = math.min(start+batchSize-1, nSample)
      -- inputs and targets
      self:sub(dv, start, stop, inplace)
      
      nSampled = nSampled + stop - start + 1
      start = start + batchSize
      collectgarbage() 
      return stop, dv
   end
end

function DataView:type(type)
   self:flush()
   self:forwardPut(self._input:type(type))
end


-------------------------------------------------
--[[ GET and Set tensor data ]]--
-------------------------------------------------
function DataView:IsFilled()
   return self._is_data_filled
end

function DataView:GetInputTensor()
   if self:IsFilled() then
      return self._input
   else
      error('not filled yet')
   end
end

function DataView:SetInputTensor(input)
   assert(torch.isTensor(input))
   self._input = input
   self._is_data_filled = true
   return self._input
end

function DataView:input(input)
   if input then
      self:SetInputTensor(input)
      return 
   end
   self.log.tracefrom('requiring views input')
   return self:GetInputTensor()
end

-----------------------------------------------
-- a generic function for transposing views from self._view
function DataView:transpose(new_view)
   local view = _.split(self._view)
   local transpositions = {}
   -- self.log.tracefrom('\t get empty')
   -- self.log.trace('calling Transpose from '..self._view..' to '..new_view)
   for i=1, #new_view do
      local j = _.indexOf(view, new_view:sub(i,i))
      if i ~= j then
         local char = view[i]
         view[i] = view[j]
         view[j] = char
         table.insert(transpositions, {j, i})
      end
   end
   return nn.Transpose(unpack(transpositions))
end
-- vim:ts=3 ss=3 sw=3 expandtab
