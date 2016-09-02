------------------------------------------------------------------------
--[[ InorderSampler ]]--
-- DataSet iterator
-- Inorder samples batches from a dataset.
-- the simplest sampler
-- two method for sampling
-- single thread: sampleEpoch(dataset)
-- multithread: sampleEpochAsync(dataset)
------------------------------------------------------------------------
local InorderSampler, parent = torch.class("dp.InorderSampler", "dp.Sampler")

function InorderSampler:__init(config)
    parent.__init(self, config)
end

------------------------------------------------------------------------
-- Build an `iterator` over samples for one epoch
-- Default is to iterate sequentially over all examples
--
-- @param dataset: dataSet which include inputDataView and outputDataView
-- @return an function object: an instance of sampler in Sampler class
--  which has member function call by (batch) and return batch, nSample, epochSize 
-- useage:
--  local sampler = dp.Sampler:sampleEpoch(dataset)
--  local batch = sampler(batch) or batch = sampler()
--
-- the return function is used recycling in the training process
------------------------------------------------------------------------
function InorderSampler:sampleEpoch(dataset)
   assert(dataset.isDataSet)
   -- dataset = dp.Sample.toDataset(dataset)
   local nSample = dataset:nSample()
   local epochSize = self._epoch_size 
   -- start index of the sample in dataset
   -- self._start = 1 -- if self._start not set, default from 1
   local nSampled = 0
   local stop_batch
   
   --[[ build iterator ]]--
   -- function which can call as: sampler(batch)
   -- return batch, min(nSampled, epochSize), epochSize
   return function(batch)
      if nSampled >= epochSize then
         return
      end
      -- i.e. the last batch may have batch_size less than self._batch_size
      -- since that there.is not enough sample, which may be dangerous
      -- TODO add fix_batch_size flags?
      stop_batch = math.min(self._start + self._batch_size - 1, nSample)
      
      -- get batch, reused the batch, if not create one 
      if batch.IsBatch then 
          assert(batch.IsFilled()) -- must be filled whtn inited
      else
         batch = dataset:InitBatchWithSize(self._batch_size)
      end

      -- batch in init already, calling sub will fill the data 
      -- into batch._inputs and batch._targets
      -- VisualDataSet:
      dataset:FillBatchOrderSample(batch, self._start, stop_batch)
      
      -- get empty indices, reuse indices tensor for resetting
      -- metadata
      batch:reset{
         batch_iter = stop_batch, -- number of examples seen so far 
         batch_size = self._batch_size,
         n_sample = stop_batch - self._start + 1, -- batch_size
         indices = batch:indices():range(self._start, stop_batch) 
         -- indices of the samples of batch in dataset 
      }
      -- data preprocesses, if no return batch 
      batch = self._ppf(batch)
      nSampled = nSampled + stop - self._start + 1

      --[[ increment self_start ]]--
      self._start = self._start + self._batch_size
      if self._start >= nSample then
         self._start = 1
      end
      self:collectgarbage()
      local epoch_stop_idx = math.min(nSampled, epochSize)
      return batch, epoch_stop_idx, epochSize
   end
end

-- used with datasets that support asynchronous iterators like ImageClassSet
-- return a function object, call by(batch, putOnly)
function InorderSampler:sampleEpochAsync(dataset)
    assert(dataset.isDataSet)
   -- dataset = dp.Sampler.toDataset(dataset)
   -- variable as control for multithreading
   local nSample = dataset:nSample()
   local epochSize = self._epoch_size or nSample
   self._start = self._start or 1
   local nSampledPut = 0
   local nSampledGet = 0
   local stop_sampleid

   local startThreadSampleBatch = function()
       if nSampledGet >= epochSize then
           -- do nothind is the Sampled Get intotal is enough for the epoch
           -- i.e. reach the end of the current epoch
           return 
       end
       -- recurrently put #epochSize sample
       -- if nSampledPut < epochSize then -- renmoved, has been Checked
       stop_sampleid = math.min(self._start + self._batch_size - 1, nSample)
       --[[ get batch]]--
       -- up values
       local uvstop_sampleid = stop_sampleid -- make it local 
       local uvbatchsize = self._batch_size
       local uvstart = self._start
       local batch = dataset:CreateBatchWithSize(batch_size)
       local callback_func = function(batch)
           -- metadata
           batch:setup
           {
               batch_iter=uvstop_sampleid, 
               batch_size=batch:nSample()
           }
           batch = self._ppf(batch)
       end
       -- ImageClassSet:subAsyncPut(batch, start, stop, callback)   
       dataset:AsyncAddOrderSampleJob(batch, self._start, stop_sampleid, callback_func) 
       -- func subAsyncPut

       nSampledPut = nSampledPut + stop_sampleid - self._start + 1
       --[[ increment self_start ]]--
       self._start = self._start + self._batch_size
       if self._start >= nSample then
           self._start = 1
       end
   end      
 
   --[[ build iterator ]]--
   local sampleBatch = function(batch)
      startThreadSampleBatch(batch)     
      batch = dataset:asyncGet()
      nSampledGet = nSampledGet + self._batch_size
      self:collectgarbage() 
      local epoch_stop_idx = math.min(nSampled, epochSize)
      return batch, epoch_stop_idx, epochSize
   end -- func samplerBatch

   assert(dataset.isAsync, "expecting asynchronous dataset")
   -- empty the async queue
   dataset:synchronize()
   -- fill task queue with some batch requests
   -- the first time call sampleEpochAsync, start 'putOnly' sampleBatch for #nThread times
   for tidx = 1, dataset.nThread do
      startThreadSampleBatch() 
   end
   return sampleBatch
end -- func sampleEpochAsync




