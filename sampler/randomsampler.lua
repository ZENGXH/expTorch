------------------------------------------------------------------------
--[[ RandomSampler ]]--
-- DataSet iterator
-- Randomly samples batches from a dataset.
------------------------------------------------------------------------
local RandomSampler, parent = torch.class("dp.RandomSampler", "dp.Sampler")

-------------------------------------------------
-- <overwrite>
-- Returns an iterator over samples for one epoch
-------------------------------------------------
function RandomSampler:sampleEpoch(dataset)
   dataset = dp.RandomSampler.toDataset(dataset)
   local nSample = dataset:nSample()
   local epochSize = self._epoch_size or nSample
   self._start = self._start or 1
   local nSampled = 0
   local stop
   
   -- build iterator
   return function(batch)
      if nSampled >= epochSize then
         return
      end
      if batch.IsBatch then 
          assert(batch.IsFilled()) -- must be filled whtn inited
      else
         batch = dataset:InitBatchWithSize(self._batch_size)
      end
      -- init a batch
      -- batch = dataset:InitBatchWithSize
      -- inputs and targets
      dataset:FillBatchRandomSample(batch, self._batch_size, nil) -- sampleFunc
      -- metadata
      batch:reset{
         batch_iter=nSampled, 
         batch_size=self._batch_size,
         n_sample=self._batch_size
      }
      batch = self._ppf(batch)
      -- batch setup part in asyncGet

      nSampled = nSampled + self._batch_size
      self._start = self._start + self._batch_size
      if self._start >= nSample then
         self._start = 1
      end
      self:collectgarbage() 
      return batch, math.min(nSampled, epochSize), epochSize
   end
end

-- used with datasets that support asynchronous iterators like ImageClassSet
function RandomSampler:sampleEpochAsync(dataset)
   dataset = dp.Sampler.toDataset(dataset)
   local nSample = dataset:nSample()
   local epochSize = self._epoch_size or nSample
   self._start = self._start or 1
   local nSampledPut = 0
   local nSampledGet = 0
     
   local startThreadSampleBatch = function()
   -- build iterator
      self.log.trace('.. Get: ', nSampledGet, ' Put: ', nSampledPut, ' epochSize ', epochSize)
      if nSampledGet >= epochSize then
         return
      end
      
      if nSampledPut < epochSize then
         -- up values
         local uvbatchsize = self._batch_size
         local uvstart = self._start
         if batch ~= nil then
            self.log.trace('batch not nil')
         else
            self.log.trace('batch is nil')
         end

       local batch = dataset:CreateBatchWithSize(batch_size)
         local callback_func = function(batch)
             -- metadata
             batch:setup{
                 batch_iter=uvstop, 
                 batch_size=batch:nSample()
             }
             batch = self._ppf(batch)
         end
       local sample_func = nil 
       dataset:AsyncAddRandomSampleJob(batch, self._batch_size, sample_func, callback_func)
         
         -- batch = self._ppf(batch)
         nSampledPut = nSampledPut + self._batch_size
         self._start = self._start + self._batch_size
         if self._start >= nSample then
            self._start = 1
         end
      end
      
   local sampleBatch = function(batch)
      startThreadSampleBatch()
         self.log.trace('not putOnly, call asyncGet')
         batch = dataset:asyncGet()
         nSampledGet = nSampledGet + self._batch_size
         self:collectgarbage() 
         return batch, math.min(nSampledGet, epochSize), epochSize
   end
   
   assert(dataset.isAsync, "expecting asynchronous dataset")
   -- empty the async queue
   dataset:synchronize()
   -- fill task queue with some batch requests
   for tidx = 1, dataset.nThread do
      log.trace('samplingBatch as thread: ', tidx)
      startThreadSampleBatch()
   end
   
   return sampleBatch
end
