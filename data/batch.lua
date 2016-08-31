------------------------------------------------------------------------
--[[ Batch ]]--
-- BaseSet subclass
-- State of a mini-batch to be fed into a model and criterion.
-- A batch of examples sampled from a dataset.
-- Serializable
------------------------------------------------------------------------
local Batch, parent = torch.class("dp.Batch", "dp.BaseSet")
Batch.isBatch = true

function Batch:__init(config)
   if not config.name then config.name='batch' end
   parent.__init(self, config)
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   local args = {} 
   dp.helper.unpack_config(args,
      {config},
      'Batch', 
      'State of a mini-batch to be fed into a model and criterion.',
      {arg='epoch_size', type='number',
       help='number of samples in original dataset'}
   )
   self._epoch_size = args.epoch_size
   parent.__init(self, config)
end

function Batch:setup(config)
   self.log.tracefrom('batch is setup in ')
   assert(type(config) == 'table', "Setup requires key-value arguments")
   local args
   args, self._batch_iter, self._batch_size, self._n_sample, 
      self._indices, self._epoch_size
      = xlua.unpack(
      {config},
      'Batch:setup', 
      'post-construction setup. Usually performed by Sampler.',
      
      {arg='batch_iter', type='number',
       help='Count of the number of examples seen so far. Used to '..
       'update progress bar. Shouldn\'t be larger than epoch_size.'}, 

      {arg='batch_size', type='number',
       help='Maximum number of examples in batch.'},

      {arg='n_sample', type='number',
       help='hardcode the number of examples'},

      {arg='indices', type='torch.Tensor', 
       help='indices of the examples in the original dataset.'},

      {arg='epoch_size', type='number', default=self._epoch_size,
       help='number of samples in epoch dataset.'}
   )
end

function Batch:batchSize()
   return self._batch_size
end

function Batch:epochSize()
   return self._epoch_size
end

function Batch:batchIter()
   return self._batch_iter
end

function Batch:indices()
   return self._indices
end

