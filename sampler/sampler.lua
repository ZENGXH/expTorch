------------------------------------------------------------------------
--[[ Sampler ]]--
-- DataSet iterator
-- Sequentially samples batches from a dataset.
-- As a sampler, it need to know the batch_size
-- one sampler can be used to sample from different dataset at the time, i.e.
-- samper if independent from dataSet
------------------------------------------------------------------------
local Sampler = torch.class("dp.Sampler")
Sampler.isSampler = true

------------------------------------------------------------------------
--[[ init Sampler ]]--
--@param config:
--  name
--  batch_size: int = 128
--  epoch_size: int = -1
--  ppf: [optional]
--  gc_freq: int = 50
------------------------------------------------------------------------
function Sampler:__init(config)
   config = config or {}
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   local args = {}
   -- batch_size, epoch_size, ppf, gc_freq = xlua.unpack(
   dp.helper.unpack_config(args,{config},
      'Sampler', 
      'Samples batches from a set of examples in a dataset. '..
      'Iteration ends after an epoch (sampler-dependent) ',
      {arg='name', type='string', default='Sampler',
       help='name of the sampler'},
      {arg='batch_size', type='number', req=true,
       help='Number of examples per sampled batches'},
      {arg='epoch_size', type='number', default=-1,
       help='Number of examples presented per epoch. '..
       'Number of examples per epoch_size, '..
       'Default is to use then entire dataset per epoch'},
      {arg='ppf', type='function', 
       help='a function that preprocesses a Batch into another Batch'},
      {arg='gc_freq', type='number', default=50,
       help='collectgarbage() every gc_freq batches'}
   )
   self.log = loadfile(paths.concat(dp.DPRNN_DIR, 'utils', 'log.lua'))()
   self.log.SetLoggerName(args.name)
   self._ppf = args.ppf or function(batch) return batch end
   self._gc_freq = args.gc_freq
   self:setBatchSize(args.batch_size)
   self._epoch_size = args.epoch_size
   self._gc_n_batch = 0
   if args.epoch_size > 0 then
      if args.batch_size > args.epoch_size then
         error("positive epoch_size should be greater than batch_size", 2)
      end
   else
      self._epoch_size = nil
   end
   log.info('[Sampler init done]')
end



------------------------------------------------------------------------
--[[ setup Sampler]]--
-- can be used to reset the batch_size?
--
--@param config: table:
--  batch_size: int
--  overwrite: bool
--  mediator: dp.Mediator
------------------------------------------------------------------------
function Sampler:setup(config)
   self.log.trace('Sampler setup')
   assert(type(config) == 'table', "Setup requires key-value arguments")
   local args = {}
   --batch_size, overwrite, mediator = xlua.unpack(
   dp.helper.unpack_config(args,{config},
      'Sampler:setup', 
      'Samples batches from a set of examples in a dataset. '..
      'Iteration ends after an epoch (sampler-dependent) ',
      {arg='batch_size', type='number',
       help='Number of examples per sampled batches'},
      {arg='overwrite', type='boolean', default=false,
       help='overwrite existing values if not nil.' .. 
       'If nil, initialize whatever the value of overwrite.'},
      {arg='mediator', type='dp.Mediator',
       help='used for communication between objects'}
   )
   if args.batch_size and (not self._batch_size or args.overwrite) then
      self:setBatchSize(args.batch_size)
   end
   self._mediator = args.mediator
end

function Sampler:setBatchSize(batch_size)
   if torch.type(batch_size) ~= 'number' or batch_size < 1 then
      error("Expecting positive batch_size get ", batch_size)
   end
   self._batch_size = batch_size
end

function Sampler:batchSize()
   return self._batch_size
end

function Sampler:report()
   return {batch_size = self._batch_size}
end

------------------------------------------------------------------------
-- static function. Checks dataset type or gets dataset from datasource
-- @param dataset, can be DataSource which contains a trainSet 
--                  or dataView 
-- convert all to dataSet format, then return
------------------------------------------------------------------------
function Sampler.toDataset(dataset)
   if dataset.isDataSource then
      -- assumes dataset is the DataSource's training set
      dataset = dataset:trainSet()
      if self then
        self._warning = true
      end
   elseif dataset.isView then
      -- assumes dataset is a set of inputs in training set
      -- create a new ['train']['input']dataset with View
      dataset = dp.DataSet{which_set='train', 
                            inputs=dataset}
   end
   assert(dataset.isDataSet, "Error : unsupported dataset type.")
   return dataset
end

function Sampler:collectgarbage()
   self._gc_n_batch = self._gc_n_batch + 1
   if self._gc_n_batch >= self._gc_freq then
      --http://bitsquid.blogspot.ca/2011/08/fixing-memory-issues-in-lua.html
      collectgarbage()
      self._gc_n_batch = 0
   end
end

------------------------------------------------------------------------
-- Build an `iterator` over samples for one epoch
-- Default is to iterate sequentially over all examples
-- @param dataset: dataSet which include inputDataView and outputDataView
-- @return an function object: an instance of sampler in Sampler class
--  which has member function call by (batch) and return batch, nSample, epochSize 
-- useage:
--  local sampler = dp.Sampler:samplerEpoch(dataset)
--  local batch = sampler(batch) or batch = sampler()
------------------------------------------------------------------------
function Sampler:sampleEpoch(dataset)
   dataset = dp.Sample.toDataset(dataset)
   local nSample = dataset:nSample()
   local epochSize = self._epoch_size or nSample
   -- start index of the sample in dataset
   self._start = self._start or 1 -- if self._start not set, default from 1
   local nSampled = 0
   local stop
   
   --[[ build iterator ]]--
   -- function which can call as: sampler(batch)
   -- return batch, min(nSampled, epochSize), epochSize
   return function(batch)
      if nSampled >= epochSize then
         return
      end
      stop = math.min(self._start + self._batch_size - 1, nSample)
      -- build up a batch given with batch_size, with [dataView]inputs and targets
      -- if batch is nil, will call batch_building with batch_size: step - self._start + 1, 
      -- in the function will call 
      -- Dataset:sub(1, 1 + stop - self._start), 
      -- i.e. the filling self.inputs[1, length] first
      
      --[[ get batch ]]--
      batch = batch or dataset:batch(stop - self._start + 1)
      -- inputs and targets
      -- batch in init already, calling sub will dill the data into batch._inputs and batch._targets
      dataset:sub(batch, self._start, stop)
      local indices = batch:indices() or torch.Tensor()      
      -- metadata
      batch:reset{
         batch_iter=stop, -- number of examples seen so far 
         batch_size=self._batch_size,
         n_sample=stop - self._start + 1, 
         indices=indices:range(self._start, stop) -- indices of the samples of batch in dataset 
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
      return batch, math.min(nSampled, epochSize), epochSize
   end
end

-- used with datasets that support asynchronous iterators like ImageClassSet
-- return a function object, call by(batch, putOnly)
function Sampler:sampleEpochAsync(dataset)
   dataset = dp.Sampler.toDataset(dataset)
   local nSample = dataset:nSample()
   local epochSize = self._epoch_size or nSample
   self._start = self._start or 1
   local nSampledPut = 0
   local nSampledGet = 0
   local stop
     
   --[[ build iterator ]]--
   local sampleBatch = function(batch, putOnly)
      if nSampledGet >= epochSize then
         return
      end
      -- recurrently put #epochSize sample
      if nSampledPut < epochSize then
         stop = math.min(self._start+self._batch_size - 1, nSample)
         --[[ get batch]]--
         -- up values
         local uvstop = stop
         local uvbatchsize = self._batch_size
         local uvstart = self._start
         -- ImageClassSet:subAsyncPut(batch, start, stop, callback)   
         dataset:subAsyncPut(batch, self._start, stop,
            function(batch) -- callback function 
               -- metadata
               batch:setup{
                   batch_iter=uvstop, 
                   batch_size=batch:nSample()
               }
               batch = self._ppf(batch)
            end)
         
         nSampledPut = nSampledPut + stop - self._start + 1

         --[[ increment self_start ]]--
         self._start = self._start + self._batch_size
         if self._start >= nSample then
            self._start = 1
         end
      end
      
      if not putOnly then
         batch = dataset:asyncGet()
         nSampledGet = nSampledGet + self._batch_size
         self:collectgarbage() 
         return batch, math.min(nSampledGet, epochSize), epochSize
      end
   end

   assert(dataset.isAsync, "expecting asynchronous dataset")
   -- empty the async queue
   dataset:synchronize()
   -- fill task queue with some batch requests
   -- the first time call sampleEpochAsync, start 'putOnly' sampleBatch for #nThread times
   for tidx=1, dataset.nThread do
      sampleBatch(nil, true)
   end
   return sampleBatch
end

-- change normal sampleEpoch to sampleEpochAsync
function Sampler:async()
   self.sampleEpoch = self.sampleEpochAsync
end
