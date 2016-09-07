----------------------------------------------------------------------
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
-- Non-multithread sampling for each epoch
-- Build an `iterator` over samples for one epoch
-- Default is to iterate sequentially over all examples
--
-- @param dataset: dataSet which include inputDataView and outputDataView
-- @return an function object: an instance of sampler in Sampler class
--  which has member function call by (batch) and return batch, nSample, epochSize 
--
-- useage:
--  local sampler = dp.Sampler:sampleEpoch(dataset)
--  local batch = sampler(batch) or batch = sampler()
--
-- the return function is used recycling in the training process
------------------------------------------------------------------------
function InorderSampler:sampleEpoch(dataset)
   assert(dataset.isDataSet)
   local nSample = dataset:nSample()
   if not self._epoch_size then 
       self.log:error('epoch size not set! set to be nSample', nSample)
       self._epoch_size = nSample 
   end
   local epochSize = self._epoch_size 
   local nSampled = 0
   local stop_sampleid
   
   --[[ build iterator ]]--
   -- function which can call as: sampler(batch)
   -- return batch, min(nSampled, epochSize), epochSize
   return function(batch)
      if nSampled >= epochSize then
         self.log:trace('nSample reach end')
         return false
      end
      -- sulotion: we need to ensure the batch_size is consistence, 
      -- decrease the self_start if there is not enough sample left
      stop_sampleid = self._start + self._batch_size - 1
      if stop_sampleid > nSample then
          stop_sampleid = nSample
          self._start = nSample - self._batch_size + 1
      end
      
      -- get batch, reused the batch, if not create one 
     if batch and batch.IsBatch then 
          assert(batch.IsFilled()) -- must be filled whtn inited
      else
         self.log:trace('\t InitBatchWithSize: ', self._batch_size)
         batch = dataset:InitBatchWithSize(self._batch_size)
      end

      -- batch in init already, calling sub will fill the data 
      -- into batch._inputs and batch._targets
      self.log:trace('\t calling FillBatchOrderSample')
      dataset:FillBatchOrderSample(batch, self._start, stop_sampleid)
      
      -- get empty indices, reuse indices tensor for resetting
      -- metadata
      batch:reset{
         batch_iter = stop_sampleid, -- number of examples seen so far 
         batch_size = self._batch_size,
         n_sample = self._batch_size, -- batch_size
         indices = batch:indices():range(self._start, stop_sampleid) 
         -- indices of the samples of batch in dataset 
      }
      -- data preprocesses, if no return batch 
      self.log:trace('\t ppf batch')
      batch = self._ppf(batch)

      --[[ increment global self_start ]]--
      nSampled = nSampled + stop_sampleid - self._start + 1
      self._start = self._start + self._batch_size
      if self._start >= nSample then
         self._start = 1
      end
      self:collectgarbage()
      local epoch_stop_idx = math.min(nSampled, epochSize)
      return batch, epoch_stop_idx, epochSize
   end
end

------------------------------------------------------------------------
-- for multithreading, batch are init in the first multithread setting
-- multithreading version of sampling iterators for one epoch
-- used with datasets that support asynchronous iterators like ImageClassSet
-- return a function object, call by(batch, putOnly)
------------------------------------------------------------------------
function InorderSampler:sampleEpochAsync(dataset)
   self.log:trace('requests iterator')
   assert(dataset.isDataSet)
   -- dataset = dp.Sampler.toDataset(dataset)
   -- variable as control for multithreading
   local nSample = dataset:nSample()
   if not self._epoch_size then 
       self.log:error('epoch size not set!', self._epoch_size, ' set to be nSample', nSample)
       self._epoch_size = nSample 
   end
   local epochSize = self._epoch_size
   local nSampledPut = 0
   local nSampledGet = 0
   local stop_sampleid
   local batch_size = self._batch_size
   -- add thread job 
   local StartThreadSampleBatch = function()
        -- do nothing if the Sampled Get intotal is enough for the epoch
       if nSampledGet >= epochSize then
           self.log:trace('nSample reach end')
           return false 
       else
           self.log:slience('get', nSampledGet ,' ', epochSize)
       end
       stop_sampleid = self._start + self._batch_size - 1
       if stop_sampleid > nSample then
           stop_sampleid = nSample
           self._start = nSample - self._batch_size + 1
       end
       --[[ get batch]]--
       -- up values
       local uvstop_sampleid = stop_sampleid -- make it local 
       local uvbatchsize = self._batch_size
       local uvstart = self._start
       -- batch should be init in AsyncAddOrderSampleJob, 
       -- #epoch_num batch will be inited then next epoch will reused them 
       -- from buffer_batch
       self.log:trace('setup callback_func')
       local callback_func = function(batch)
           -- metadata
           batch:setup {
               batch_iter=uvstop_sampleid, 
               batch_size=batch_size
           }
           batch = self._ppf(batch)
       end
       self.log:trace('calling AsyncAddOrderSampleJob: ')
       dataset:AsyncAddOrderSampleJob(batch, self._start, stop_sampleid, 
          callback_func)
       --[[ increment self_start ]]--
       nSampledPut = nSampledPut + stop_sampleid - self._start + 1
       self._start = self._start + self._batch_size
       if self._start >= nSample then
           self._start = 1
       end
       return true
   end      
   --[[ build iterator ]]--
   local sampleBatch = function(batch)
       local batch_left = StartThreadSampleBatch()
      if not batch_left then
          return
      end
      batch = dataset:asyncGet()
      nSampledGet = nSampledGet + self._batch_size
      self:collectgarbage() 
      -- local epoch_stop_idx = math.min(nSampledGet, epochSize)
      return batch, nSampledGet, epochSize
   end -- func samplerBatch

   assert(dataset.isAsync, "expecting asynchronous dataset")
   -- empty the async queue
   -- pull all batch from _recv_batches input buffer_batch
   dataset:synchronize()
   -- fill task queue with some batch requests
   for tidx = 1, dataset.nThread do
      StartThreadSampleBatch() 
   end
   return sampleBatch
end -- func sampleEpochAsync




