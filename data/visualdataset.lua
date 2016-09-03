------------------------------------------------------------------------
--[[ VisualDataSet ]]--
-- for common function of VisualDataSet and videoclassset
-- interface
------------------------------------------------------------------------
local VisualDataSet, parent = torch.class("dp.VisualDataSet", "dp.DataSet")
VisualDataSet.isVisualDataSet = true
function VisualDataSet:__init(config)
   parent.__init(self, config)
   -- buffers
   self._imgBuffers = {}
end

function VisualDataSet:BuildIndex()
   error('not implement')
end

function VisualDataSet:SaveIndex()    
   error('not implement')
end

function VisualDataSet:LoadIndex()
   error('not implement')
end

------------------------------------------------------------------------
-- create a batch with xxViewInput and ClassViewTargets
-- with batchSize
-- @param batch_size
-- 
-- @return functio object, take `batch` as input, output a dp.Batch
------------------------------------------------------------------------

--[[ factory ]]
function VisualDataSet:FillBatchWithSize(batch, batch_size)
   error('confused function, removed')
end

------------------------------------------------------------------------
-- create a batch with 'which_set' and 'epoch_size'
-- fill the batch with input View and target View with batch_size shape
-- if the batch has been create but not input View and target View set, please
-- call : FillBatchWithSize
-- <useage> you can either CreateEmptyBatchIfNil(DEPRECATED) or
-- InitBatchWithSize(batch_size)
-- @batch_size: int
--
-- @return: batchsize inited wich 
------------------------------------------------------------------------
function VisualDataSet:InitBatchWithSize(batch_size)
   assert(torch.type(batch_size) == 'number' and batch_size > 0)
   local batch = self:CreateEmptyBatchIfNil()
   assert(batch.isBatch and not batch:IsFilled())
   self.log.tracefrom('request batch with size ', batch_size, 
   ' sample_size(input) ', unpack(self._input_shape_set))
   batch:SetView('input', dp[self._input_view_type](self._input_shape, 
   torch[self._input_view_tensor](batch_size, unpack(self._input_shape_set))))
   if 0 == #self._target_shape_set then
      batch:SetView('targets', dp[self._target_view_type](self._output_shape, torch[self._target_view_tensor](batch_size)))
   else
      batch:SetView('targets', dp[self._target_view_type](self._output_shape, torch[self._target_view_tensor](batch_size, unpack(self._target_shape_set))))
   end
   return batch
end
------------------------------------------------------------------------
-- nSample(), nSample(class)
-- return number of sample in the dataset
--   or number of sample in some class required by className or classIndices
-- @param class: string [optional]
-- @param list:
-- 
-- @return sample number: Int
------------------------------------------------------------------------
--[[overwrite]]--
function VisualDataSet:nSample(class, list)
   list = list or self.classList
   if not class then
      return self._n_sample
   elseif type(class) == 'string' then
      return list[self._classIndices[class]]:size(1)
   elseif type(class) == 'number' then
      return list[class]:size(1)
   end
end

function VisualDataSet:getImageBuffer(i)
   return self:GetImageBuffer(i)
end

function VisualDataSet:GetImageBuffer(i)
   error('not implement here, buffer has tensor in size based on sub-class')
   self._imgBuffers[i] = self._imgBuffers[i] or torch.FloatTensor()
   return self._imgBuffers[i]
end

function VisualDataSet:classes()
   return self._classes
end

function VisualDataSet:loadImage(path)
   error('not implement')
end

------------------------------------------------------------------------
-- <abstract> do the combining of tensor in to a batch in the subclass since
-- then the output and input shape may be different
-- @param inputTable
-- @orarm outputTable
-- @oaram inputTensor
-- @oaram outputTensor
--
-- @return inputTensor
-- @return outputTensor
------------------------------------------------------------------------
function VisualDataSet:tableToTensor(...)
   error('not implement')
end

------------------------------------------------------------------------
-- init multithread in worker, set up dataset in each worker,
-- set up a batch for each dataset with filled
--
-- @param nThread: int
------------------------------------------------------------------------

function VisualDataSet:multithread(nThread,batch_size)
   self.log.info('get multithread: ', nThread, ' b ', batch_size)
   assert(nThread > 0 and  batch_size > 0)
   local nThread = nThread or 2
   assert(batch_size and batch_size > 0, 'batch_size required')
   if not paths.filep(self._cache_path) then
      -- workers will read a serialized index to speed things up
      self:saveIndex()
   end

   local mainSeed = os.time()
   local config = self._config
   config.cache_mode = 'readonly'
   config.verbose = self._verbose 
   local threads = require "threads"
   local batch_size = batch_size
   threads.Threads.serialization('threads.sharedserialize')
   self.log.info('init threads with dataset: ', self._class_set)

   self._threads = threads.Threads(
   nThread,
   -- all function below will be executed in all thread
   function() -- make a separated f1 containing all the definitions 
      -- print('threading')
      require 'dprnn.dprnn'
   end,
   function(idx) -- other code in f2
      opt = options -- pass to all donkeys via upvalue
      tid = idx
      local seed = mainSeed + idx
      math.randomseed(seed)
      torch.manualSeed(seed)
      self.log.info(string.format('Starting worker thread with id: %d seed: %d', tid, seed), 
        'setup class_set: ', self._class_set, ' with batch_size ', batch_size)
      dataset = dp[self._class_set](config)
      tbatch = dataset:InitBatchWithSize(batch_size)
   end
   )
   self._send_batches = dp.Queue() -- batches sent from main to threads
   self._recv_batches = dp.Queue() -- batches received in main from threads
   self._buffer_batches = dp.Queue() -- buffered batches

   -- public variables
   self.nThread = nThread
   self.isAsync = true
   -- #TODO : init _buffer_batches with size nThread
end

------------------------------------------------------------------------
-- call by InorderSampler, given batch created with batch_size,
-- sample at start, for #nSample, update the self._start
-- send request to worker : put request into queue

-- @param batches
-- @param start, stop: int, for imageclassset, it is the index for the
--          imagePaths, for videoclassset, it is the index for the video
--          for order sampling, the start is remembered by sampler, since that
--          there different dataset working in different thread
-- @param callback
------------------------------------------------------------------------
function VisualDataSet:AsyncAddOrderSampleJob(batch, start, stop, callback)   
   if not batch or (batch and batch,isBatch and batch.IsFilled()) then
      -- buffer_batches are filled in :synchronize() by batch in _recv_batches
      batch = (not self._buffer_batches:empty() and self._buffer_batches:get()) or self:InitBatchWithSize(self._batch_size)
   end
   assert(batch.isBatch and batch.IsFilled())
   local input = batch:GetView('input'):GetInputTensor()
   local target = batch:GetView('target'):GetInputTensor()
   assert(input and target) 
   self._send_batches:put(batch)
   -- the job callback (runs in data-worker thread)
   local worker_job =  function() 
      -- tbatch is a (empty) batch container work in current thread
      -- put not inputTensor and targetTensor input tBatch container
      -- call batchFill function, i.e. the acture sample function
      -- to update the inputTensor and targetTensor, put return it to
      -- the mainThread
      tbatch:GetView('input'):forwardPut(self._input_shape, input)
      tbatch:GetView('target'):forwardPut(self._output_shape, target)
      dataset:FillBatchOrderSample(tbatch, start, stop)
      return input, target
      -- the callback return one ore many values which will be 
      -- serialized and unserialized as arguments to the endcallback function. 
   end
   -- the endcallback (runs in the main thread)
   local main_thread_job = function(input, target) 
      -- feed the input and target into 
      -- pull batch from send batch buffer, reset the input, push to recv batch
      -- buffer
      local batch = self._send_batches:get()
      -- filling input data
      batch:GetView('input'):forwardPut(self._input_shape, input)
      batch:GetView('target'):forwardPut(self._output_shape, target)
      -- init call batch:setup and do preprocesses 
      callback(batch)
      batch:GetView('target'):setClasses(self._classes)
      self._recv_batches:put(batch)
   end

   self._threads:addjob(worker_job, main_thread_job)
end

function VisualDataSet:AsyncAddRandomSampleJob(batch, batch_size, sample_func, callback)
   local input = batch:GetView('input'):GetInputTensor()
   local target = batch:GetView('target'):GetInputTensor()

   local p = torch.pointer(input:storage()) 
   -- transfer the storage pointer over to a thread
   local targetPointer = tonumber(ffi.cast('intptr_t', 
   torch.pointer(target:storage())))

   self.log.trace('get target pointer')
   local inputPointer = tonumber(ffi.cast('intptr_t', 
   torch.pointer(input:storage())))
   self.log.trace('get input pointer')
   input:cdata().storage = nil
   target:cdata().storage = nil

   self._send_batches:put(batch)

   self.log.trace('put batch')
   assert(self._threads:acceptsjob())
   self.log.trace('start add job')
   local view = self._input_shape

   -- the job callback (runs in data-worker thread)
   local worker_job = function()
      -- set the transfered storage
      torch.setFloatStorage(input, inputPointer)
      torch.setIntStorage(target, targetPointer)
      tbatch:inputs():forward(view, input)
      tbatch:targets():forward('b', target)
      -- fill tbatch wilth inputData and targetData         
      dataset:FillBatchRandomSample(tbatch, nSample)
      assert(tbatch:inputs():input()) 
      assert(tbatch:targets():input()) 
      -- transfer it back to the main thread
      local istg = tonumber(ffi.cast('intptr_t', 
      torch.pointer(input:storage())))
      local tstg = tonumber(ffi.cast('intptr_t', 
      torch.pointer(target:storage())))
      input:cdata().storage = nil
      target:cdata().storage = nil
      return input, target, istg, tstg
   end  

   -- the endcallback (runs in the main thread)
   local main_thread_job = function(input, target, istg, tstg)
      local batch = self._send_batches:get()
      torch.setFloatStorage(input, istg)
      torch.setIntStorage(target, tstg)
      batch:inputs():forward('btchw', input)
      batch:targets():forward('b', target)
      callback(batch) 
      --callback: setup batch and run sampler's ppf on batch, view may be changed
      batch:targets():setClasses(self._classes)
      -- self.log.trace('putting to _recv_batches: ', #self._recv_batches)
      self._recv_batches:put(batch)
   end  

   self._threads:addjob(worker_job, main_thread_job)

end

function VisualDataSet:FillBatchOrderSample(batch, start, stop, callback)
   error('not implement, VisualDataSet can not be worker')
end

function VisualDataSet:FillBatchRandomSample(batch, nSample, callback, sample_func)
   error('not implement, VisualDataSet can not be worker')
end
-- pull batches from self._recv_batches and push to _buffer_batches
function VisualDataSet:synchronize()
   self._threads:synchronize()
   while not self._recv_batches:empty() do
      self._buffer_batches:put(self._recv_batches:get())
   end
end

-- recv results from worker : get results from queue
function VisualDataSet:asyncGet()
   -- necessary because Threads:addjob sometimes calls dojob...
   self.log.info('asyncGet is called')
   if self._recv_batches:empty() then
      self._threads:dojob()
   end
   return self._recv_batches:get()
end


-- vim:ts=3 ss=3 sw=3 expandtab
