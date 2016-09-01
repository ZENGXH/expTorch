------------------------------------------------------------------------
--[[ Sampler ]]--
-- DataSet iterator, abstract
-- Sequentially samples batches from a dataset.
-- As a sampler, it need to know the batch_size
-- one sampler can be used to sample from different dataset at the time, i.e.
-- samper if independent from dataSet
--
-- difference between sampleEpoch and sampleEpochAsync
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


function Sampler:sampleEpoch(dataset)
    self.log.fatal('DEPRECIATED without Implemented, Sampler is abstract, use InorderSampler instead')
    return dp.InorderSampler.sampleEpoch(self, dataset)
end

function Sampler:sampleEpochAsync(dataset)
    self.log.fatal('DEPRECIATED without Implemented, Sampler is abstract, use InorderSampler instead')
    return dp.InorderSampler.sampleEpochAsync(self, dataset)
end
-- change normal sampleEpoch to sampleEpochAsync
function Sampler:async()
   self.sampleEpoch = self.sampleEpochAsync
end
