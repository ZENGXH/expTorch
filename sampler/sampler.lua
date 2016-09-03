------------------------------------------------------------------------
--[[ Sampler ]]--
-- DataSet iterator, <abstract>
-- for the real dataset iterator, need to define sampleEpoch and sampleEpochAsync
--
-- Sequentially samples batches from a dataset.
-- As a sampler, it need to know the batch_size
-- one sampler can be used to sample from different dataset at the time, i.e.
-- sampler if independent from dataSet
--
-- difference between `sampleEpoch` and `sampleEpochAsync`
-- knowing the start and end index of the data in the dataSet
-- `sampleEpoch` will call `dataset:sub` to get the data and filled then into the
-- batch in iterators
--
-- `sampleEpochAsync` will submit many jobs when it is first inited, 
-- which are calling `dataset:subAsyncPut` 
-- without putting data into the batch in the iterator
-- TODO add fix_batch_size flags?
------------------------------------------------------------------------
local Sampler = torch.class("dp.Sampler")
Sampler.isSampler = true

------------------------------------------------------------------------
--[[ init Sampler ]]--
--@param config:
--  name
--  batch_size: int = 128, number of samples in a batch, use for request 
--      for batch from dataset 
--  epoch_size: int = -1, number of samples in a epoch, -1 means the 
--      whole data set size
--  ppf: [optional] preprocesser
--  gc_freq: int = 50
--  overwrite bool,??
--  mediator
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
       help='collectgarbage() every gc_freq batches'},
      -- {arg='overwrite', type='boolean', default=false,
      -- help='overwrite existing values if not nil.' .. 
      -- 'If nil, initialize whatever the value of overwrite.'},
      {arg='mediator', type='dp.Mediator',
       help='used for communication between objects'}
 
   )
   self.log = loadfile(paths.concat(dp.DPRNN_DIR, 'utils', 'log.lua'))()
   self.log.SetLoggerName(args.name)

   self._ppf = args.ppf or function(batch) 
        return batch 
    end
   self._gc_freq = args.gc_freq
   self:ResetBatchSize(args.batch_size)
   self._epoch_size = (args.epoch_size > 0 and args.epoch_size) or nil
   self._gc_n_batch = 0
   if args.epoch_size > 0 then
      if args.batch_size > args.epoch_size then
         error("positive epoch_size should be greater than batch_size", 2)
      end
   else
      self._epoch_size = nil
   end
   self._mediator = args.mediator
   log.info('[Sampler init done]')
   self._start = 1 -- init with 1
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

------------------------------------------------------------------------
-- <abstract> methof to sample for one epoch's data, 
-- sampleEpochAsync for the sampling of dataset  which support 
-- multithreading 
------------------------------------------------------------------------
function Sampler:sampleEpoch(dataset)
    error('without Implemented, Sampler is abstract, use InorderSampler instead')
end

function Sampler:sampleEpochAsync(dataset)
    error('without Implemented, Sampler is abstract, use InorderSampler instead')
end


function Sampler:report()
   return {batch_size = self._batch_size}
end

------------------------------------------------------------------------
-- get and set the batch_size and epoch_size
------------------------------------------------------------------------
function Sampler:ResetBatchSize(batch_size)
   assert(torch.type(batch_size) ==  'number', batch_size)
   assert(batch_size > 0, 'get batch_size '..tostring(batch_size))
   self._batch_size = batch_size
end

function Sampler:GetBatchSize()
    return self._batch_size
end

function Sampler:ResetEpochSize(epoch_size)
   assert(torch.type(epoch_size) ==  'number', epoch_size)
    self._epoch_size = expoch_size
end

function Sampler:GetEpochsize()
    return self._epoch_size
end

function Sampler:collectgarbage()
   self._gc_n_batch = self._gc_n_batch + 1
   if self._gc_n_batch >= self._gc_freq then
      --http://bitsquid.blogspot.ca/2011/08/fixing-memory-issues-in-lua.html
      collectgarbage()
      self._gc_n_batch = 0
   end
end


-- change normal sampleEpoch to sampleEpochAsync
function Sampler:async()
   self.sampleEpoch = self.sampleEpochAsync
end

function Sampler:setup(config)
  self.log.tracefrom('')
  self.log.fatal('depreciate') 
  self.__init(config) -- redirect
end

function Sampler:setBatchSize(batch_size)
   self.log.fatal('depreciated')
   return self:ResetBatchSize(batch_size)
end
function Sampler:batchSize()
   self.log.fatal('depreciated')
   return self:GetBatchSize()
end

function Sampler:SetMediator(m)
    assert(m.isMediator)
    self._mediator = m
end
