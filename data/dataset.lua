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

function DataSet:ioShapes(input_shape, output_shape)
   if input_shape or output_shape then
      self._input_shape = input_shape or self._input_shape
      self._output_shape = output_shape or self._output_shape
      return
   end
   local iShape = self._input_shape or self:inputs() and self:inputs():view()
   local oShape = self._output_shape or self:targets() and self:targets():view()
   assert(iShape and oShape, "Missing input or output shape")
   return iShape, oShape
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

function DataSet:CreateBatchWithSize(batch_size)
    assert(batch_size > 0)
    local batch = self:CreateEmptyBatchIfNil()
    return self:FillBatchWithSize(batch, batch_size)
end

function DataSet:CreateBatchWithSub(start, stop)
    assert(start > 0 and stop > start)
    local batch = self:CreateEmptyBatchIfNil()
    return self:FillBatchWithSub(batch, start, stop)
end

function DataSet:CreateBatchWithindex(indices)
    assert(torch.isTensor(indices))
    local batch = self:CreateEmptyBatchIfNil()
    return self:FillBatchWithIndex(batch, indices)
end

function DataSet:FillBatchWithSize(batch, batch_size)
    assert(batch.isBatch)
    batch:SetView('input', self:inputs():sub(1, batch_size))
    if self:targets() then
        batch:SetView('target', self:targets():sub(1, batch_size))
    end
    return batch
end

function DataSet:FillBatchWithSub(batch, start, stop)
    assert(batch.isBatch, "Expecting dp.Batch at arg 1")
    assert(batch:IsFilled())
    batch:SetView('input', 
            self:GetView('input'):CreateSubViewWithSub(start, stop))
    if self:targets() then
        batch:SetView('target', 
            self:GetView('target'):CreateSubViewWithSub(start, stop))
    end
    return batch  
end

function DataSet:FillBatchWithIndex(batch, indices)
    assert(batch.isBatch, "Expecting dp.Batch at arg 1")
    assert(batch:IsFilled())
    batch:SetView('input', 
            self:GetView('input'):CreateBatchWithIndex(indices))
    if self:targets() then
        batch:SetView('target', 
            self:GetView('target'):CreateBatchWithIndex(indices))
    end
    return batch  
end

function DataSet:batch(batch_size)
   self.log.trace('calling batch with batch_size: ', batch_size)
   return self:CreateBatchWithSize(batch_size)
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

function DataSet:sub(batch, start, stop)
   if (not batch) or (not stop) then 
      -- batch not given or stop not given
      if batch then
         -- only receive two arguments
         stop = start
         start = batch
      end
      self.log.trace('building batch with size ', self:nSample())
      -- get a DataView Contains sub_data from start to stop of the orignal inputs
      return self:CreateBatchWithSub(start, stop)
  end
   self.log.trace('dataset: sub from ', start, ' to ', stop)
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
function DataSet:index(batch, indices)
   if (not batch) or (not indices) then 
      indices = indices or batch
      return self:CreateBatchWithIndex(indices) 
   end

   return self:FillBatchWithIndex(batch, indices)

end
