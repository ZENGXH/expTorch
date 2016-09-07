------------------------------------------------------------------------
--[[ DataSet ]]--
-- BaseSet subclass
-- Contains inputs and optional targets. Used for training or
-- evaluating a model. Inputs and targets are tables of Views.

-- Unsupervised Learning :
-- If the DataSet is for unsupervised learning, only inputs need to 
-- be provided.

-- Multiple inputs and outputs :
-- Inputs and targets should be provided as instances of 
-- dp.View to support conversions to other axes formats. 
-- Inputs and targets may also be provided as dp.ListTensors. 
-- Allowing for multiple targets is useful for multi-task learning
-- or learning from hints. In the case of multiple inputs, 
-- images can be combined with tags to provided for richer inputs, etc. 
--
-- method:
--   support return ioShapes of the input and output View
--
------------------------------------------------------------------------
local DataSet, parent = torch.class("dp.DataSet", "dp.BaseSet")
DataSet.isDataSet = true

function DataSet:__init(config)
    parent.__init(self, config)
end
------------------------------------------------------------------------
-- builds a batch (factory method) for input and output DataSet
-- @param batch_size: int, size of the batch
-- reuses the inputs and targets (so don't modify them)
-- @return Batch instances with batch_size
------------------------------------------------------------------------
function DataSet:CreateEmptyBatchIfNil(batch)
     if batch then return batch end
     return dp.Batch{
        which_set=self:whichSet(), 
        epoch_size=self:nSample()
    }
end

function DataSet:CreateBatchWithindex(indices)
    error('depreciated')
end

function DataSet:InitBatchWithSize(batch_size)
    error('depreciated')
end

----------------------------------------------------------------------
-- fill the batch with different method, the batch should be create or
-- filled(given the batch_size) outside
--
-- @param batch, inited and filled
-- @param batch_size, for random sapling
-- @param start, stop: range of the sample Id for in order sample
-- @param indices 
--
-- @return batch with Data
----------------------------------------------------------------------


function DataSet:FillBatchWithSub(batch, start, stop)
    error('not implementm different for load once or not data')
end

function DataSet:FillBatchWithIndex(batch, indices)
    error('not implementm different for load once or not data')
end

------------------------------------------------------------------
-- get/set input and out shape
-- for imageDataSet and VideoDataSet they have _input_shape originally
-- if reset may ??
------------------------------------------------------------------
function DataSet:SetOutputShape(output_shape)
    self.log:info('orignal shape in ', self._input_shape, ' out: ', self.output_shape)
    self._output_shape = output_shape or self._output_shape
    self.log:info('changed into: shape in ', self._input_shape, ' out: ', self.output_shape)
end

function DataSet:SetInputShape(input_shape)
    self.log:info('orignal shape in ', self._input_shape, ' out: ', self.output_shape)
    self._input_shape = input_shape or self._input_shape
    self.log:info('changed into: shape in ', self._input_shape, ' out: ', self.output_shape)
end

function DataSet:GetInputShape()
    local iShape = self._input_shape or self:GetView('input') and self:GetView('input'):view()
    assert(iShape, "Missing input or output shape")
    return iShape
end

function DataSet:GetOutputShape()
    local oShape = self._output_shape or self:GetView('target') and self:GetView('target'):view()
    assert(oShape, "Missing input or output shape")
    return oShape
end

---------------------------------------------------------------------------
-- reuses the inputs and targets (so don't modify them)
-- given 1 and batch_size: builds a batch with inputsView and outputView in
--   size batch_size
-- given batch, s, e: fill the batch by calling `sub` of th inputsView and
-- outputViews act as a driver 
--
-- @param batch_size: int, size of the batch
-- @return Batch instances with batch_size, dataView filled or unfilled
---------------------------------------------------------------------------

-- DEPRECATED
-- used FillBatchWithSub and CreateEmptyBatchIfNil if needed
function DataSet:sub(batch, start, stop)
   if (not batch) or (not stop) then 
      -- batch not given or stop not given
      if batch then
         -- only receive two arguments
         stop = start
         start = batch
      end
      self.log:trace('building batch with size ', self:nSample())
      -- get a DataView Contains sub_data from start to stop of the orignal inputs
    assert(start > 0 and stop > start)
    local batch = self:CreateEmptyBatchIfNil()
    return self:FillBatchWithSub(batch, start, stop)
  end
   self.log:trace('dataset: sub from ', start, ' to ', stop)
   return self:FillBatchWithSub(batch, start, stop)
end

---------------------------------------------------------------------------
-- reuses the inputs and targets (so don't modify them)
-- given 1 and index: builds a batch with inputsView and outputView which is
--  build by indexing from the inputs and outputs size
-- given batch, s, e: fill the batch by calling `sub` of th inputsView and
-- outputViews act as a driver 
--
-- @param batch: 
-- @param indices:
--
-- @return Batch instances, dataView filled or unfilled
---------------------------------------------------------------------------
-- DEPRECATED
-- use FillBatchWithIndex and CreateEmptyBatchIfNil instead
function DataSet:index(batch, indices)
   if (not batch) or (not indices) then 
      indices = indices or batch
   end

   assert(torch.isTensor(indices))
   local batch = self:CreateEmptyBatchIfNil(batch)
   return self:FillBatchWithIndex(batch, indices)
end

-- DEPRECATED
function DataSet:ioShapes(input_shape, output_shape)
   if input_shape or output_shape then
       self:SetOutputShape(output_shape)
       self:SetInputShape(input_shape)
   end
   return self:GetInputShape(), self:GetOutputShape()
end

function DataSet:FillBatchWithSize(batch, batch_size)
    error('not implementm different for load once or not data')
end

function DataSet:batch(batch_size)
   self.log:trace('calling batch with batch_size: ', batch_size)
    assert(batch_size > 0)
    return self:InitBatchWithSize(batch_size)
end
