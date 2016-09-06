------------------------------------------------------------------------
--[[ Propagator ]]--
-- Abstract Class for propagating a sampling distribution (Sampler) 
-- through a module in order to evaluate its criterion, train, etc.
-- To make your own training algorithm, you can build your own 
-- propagator. If you can reduce it to reusable components, you could 
-- then refactor these into visitors, observers, etc.
-- typeconvertsion:
--  if use cuda, the Propagator will do type conversion in forward 
--  and backward
------------------------------------------------------------------------
local Propagator = torch.class("dp.Propagator")
Propagator.isPropagator = true

function Propagator:__init(config)   
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   local args = {} 
   dp.helper.unpack_config(args, {config}, 'Propagator', 
      'Propagates Batches sampled from a DataSet using a Sampler '..
      'through a Model in order to evaluate a Loss, provide Feedback '.. 
      'or train the model',
      {arg='name', type='string', default='propagator', 
      help='name of the Propagator'},
      {arg='loss', type='nn.Criterion',
       help='a neural network Criterion to evaluate or minimize'},
      {arg='callback', type='function',
       help='function(model, report) that does things like'..
       'update model, gather statistics, decay learning rate, etc.'},
      {arg='epoch_callback', type='function', 
       help='function(model, report) that is called between epochs'},
      {arg='sampler', type='dp.Sampler', 
       help='Iterates through a DataSet. [Default=dp.Sampler()]'},
      {arg='observer', type='dp.Observer', 
       help='observer that is informed when an event occurs.'},
      {arg='feedback', type='dp.Feedback',
       help='takes predictions, targets, model and visitor as input '..
       'and provides feedback through report(), setState, or mediator'},
      {arg='progress', type='boolean', default=false, 
       help='display progress bar'},
      {arg='verbose', type='boolean', default=true,
       help='print verbose information'},
      {arg='stats', type='boolean', default=false,
       help='display performance statistics (speed, etc). '..
      'Only applies if verbose is true.'},
      {arg='n_display_interval', type='number', default=100, 
       help='display interval for experiment information in monitor'}
   )
   self.tensorType = 'torch.FloatTensor' -- default 
   self.log = loadfile(paths.concat(dp.DPRNN_DIR, 'utils', 'log.lua'))()
   self.log.SetLoggerName(args.name)
   self:sampler(args.sampler or dp.Sampler())
   self:loss(args.loss) -- setup Criterion
   self:observer(args.observer)
   self:feedback(args.feedback)
   self:callback(args.callback)
   self:epochCallback(args.epoch_callback or function() return end)
   self._progress = args.progress
   self._verbose = args.verbose
   self._stats = args.stats
   self.n_batch = 0 -- global as counter for display
   self.n_display_interval = args.n_display_interval
end

function Propagator:setup(config)
   assert(type(config) == 'table', "Setup requires key-value arguments")
   local args = {}
   -- id, model, mediator, target_module = xlua.unpack(
   dp.helper.unpack_config(args,{config},
      'Propagator:setup', 
      'Post-initialization setup of the Propagator',
      {arg='id', type='dp.ObjectID', req=true,
       help='uniquely identifies the propagator.'},

      {arg='model', type='nn.Module',
       help='the model that is to be trained or tested',},

      {arg='mediator', type='dp.Mediator', req=true,
       help='used for inter-object communication.'},

      {arg='target_module', type='nn.Module', 
       help='Optional module through which targets can be forwarded'}
   )
   local id = args.id
   local model = args.model
   local mediator = args.mediator
   local target_module = args.target_module
   assert(torch.isTypeOf(id, 'dp.ObjectID'))
   assert(torch.isTypeOf(args.mediator, 'dp.Mediator'))
   self._id = args.id
   self._mediator = args.mediator
   self:model(args.model)
   self._target_module = args.target_module or nn.Identity()
   self._sampler:SetMediator(mediator)
   -- self._sampler:setup{mediator=mediator, model=model}
   
   if self._observer then 
       self._observer:setup{mediator=mediator, subject=self} 
   end

   if self._feedback then 
       self._feedback:setup{mediator=mediator, 
            propagator=self, 
            dataset=dataset} 
   end
   
   if self._visitor then 
       self._visitor:setup{mediator=mediator, 
            model=self._model, 
            propagator=self} 
   end
end

-- create sampler for current dataset, 
-- pass model and report to self._epoch_callback()
-- call propagateBatch recurrently
function Propagator:propagateEpoch(dataset, report)
   self.log.trace('propagating: ')
   self.sumErr = 0
   if self._feedback then
      self.log.trace('reset feedback ')
      self._feedback:reset()
   end
   -- local vars
   local start_time = sys.clock()
   local batch, i, n, last_n
   self.n_batch = 0
   if self._stats then
      print('==> epoch # '..(report.epoch + 1)..
        ' for '..self:name()..' :')
   end

   if self._model.forget then
       self.log.trace('calling forget')
      -- for recurrent modules, forget between epochs
      self._model:forget()
   end
   self._epoch_callback(self._model, report)
   -- create an sampler object in class Sampler
   self.log.trace('set up sampler interator: ')
   local sampler = self._sampler:sampleEpoch(dataset)
   if not report or report == nil then
       report = {}
   end
   while true do
      -- reuse the batch object
      -- if not exist then sampler(batch) will create one
      if batch then
         assert(torch.type(batch) == 'dp.Batch')
      end
      batch, nSampleRun, epochSize = sampler(batch)
      if not batch then -- fail?
         -- for aesthetics :
         if self._progress then
            xlua.progress(last_n, last_n)
         end
         break 
      end
      self.nSample = nSampleRun -- == min(nSampled, epochSize)
      --    forward + monitor + [backward + ]callback + doneBatch
      self:propagateBatch(batch, report)
      
      if self._progress then
         xlua.progress(nSampleRun, epochSize)
      end
      last_n = epochSize
      self.n_batch = self.n_batch + 1
   end
   -- time taken
   self._epoch_duration = sys.clock() - start_time
   self._batch_duration = self._epoch_duration / math.max(self.n_batch, 0.000001)
   self._example_speed = last_n / self._epoch_duration
   self._batch_speed = self.n_batch / self._epoch_duration
   if self._stats and self._verbose then
      self.log.info("==> example speed = "..self._example_speed..' examples/s')
   end
end      
---------------------------------------------------------------
-- calling forward(batch)
-- monitor(batch, report)
-- [backward(batch)] -- optimizer
-- callback(mdoel, report)
-- doneBatch(report)
---------------------------------------------------------
function Propagator:propagateBatch(batch)
   error"NotImplementedError"
end

function Propagator:forward(batch)
   local input = batch:GetView('input'):GetInputTensor()
   local target = batch:GetView('target'):GetInputTensor()
   -- type conversion, corporate with :type(type)
   input = input:type(self.tensorType)
   target = target:type(self.tensorType)
   target = self._target_module:forward(target)
   if self._include_target then
       input = {input, target}
   end
   -- useful for calling accUpdateGradParameters in callback function
   self._model.dpnn_input = input
   -- forward propagate through model
   self.output = self._model:forward(input)
   assert(self.output)
   if not self._loss then
      return
   end
   self._loss = self._loss
   -- measure loss
   self.log.slience('Propagator forward: input, target', 
      input:type(), target:type())
   self.log.slience('Pceropagator forward: output', self.output)
   self.err = self._loss:forward(self.output, target)
end

----------------------------------------------------------
-- add to feedback
-- self.sumErr, accumulated error over all the batch sent to monitor
-- publish to mediator
-- @param batch
-- @param report: table
-----------------------------------------------------------
function Propagator:monitor(batch, report)
   if not report or report == nil then
       report = {}
   end
   self.log.trace('monitoring')
   self.sumErr = self.sumErr + (self.err or 0)
   -- feed batch need to be called before observer 
   if self._feedback then
       -- call _add of the read feedback object
      self._feedback:add(batch, self.output, report)
   end
   if self._observer then
      self._observer:doneBatch(report)
   end
   --publish report for this optimizer
   -- [[ add observer ?]]
   self._mediator:publish(self:name()..':'.."doneFeedback", report, batch)
end

function Propagator:doneBatch(report)
   --publish report for this optimizer
   self._mediator:publish(self:name()..':'.."doneBatch", report)
end

---------------------------------------------------------------------------
--[[ Epoch report ]]--
-- returns a log for the current epoch, in the format of a table
-- or we could create an EpochLog class to help with this.
-- But a table is more flexible. The advantage over pylearn2 is that 
-- the log of an epoch is structured, as opposed to just a list of 
-- channel names and values. Furthermore, values can be anything 
-- serializable.
---------------------------------------------------------------------------
function Propagator:report()
   local avgErr
   if self._loss and self.sumErr and self.nSample > 0 then
      avgErr = self.sumErr / self.nSample
      if self._verbose then
         -- self:id():toString()
         self.log.info(string.format('epoch loss avgErr %.4f', avgErr))
      end
   end
   local report = {
      name = self:id():name(),      
      loss = self._loss and self._loss.report and self._loss:report() or avgErr,
      sampler = self._sampler:report(),
      epoch_duration = self._epoch_duration,
      batch_duration = self._batch_duration,
      example_speed = self._example_speed,
      num_batches = self._num_batches,
      batch_speed = self._batch_speed
   }
   if self._feedback then
      report.feedback = self._feedback:report()
   end
   return report
end

function Propagator:sampler(sampler)
   if sampler then
      self._sampler = sampler
      return
   end
   return self._sampler
end

function Propagator:id(id)
   if id then
      self._id = id
      return
   end
   return self._id
end

function Propagator:name()
   return self._id:name()
end

function Propagator:model(model)
   if model then
      self._model = model
      return
   end
   return self._model
end

function Propagator:observer(observer)
   if observer then
      if torch.type(observer) == 'table' then
         --if list, make composite observer
         observer = dp.CompositeObserver(observer)
      end
      self._observer = observer
      return
   end
   return self._observer
end

function Propagator:callback(callback)
   if callback then
      assert(torch.type(callback) == 'function', "expecting function")
      self._callback = callback
      return
   end
   return self._callback
end

function Propagator:epochCallback(callback)
   if callback then
      assert(torch.type(callback) == 'function', "expecting function")
      self._epoch_callback = callback
      return
   end
   return self._epoch_callback
end

function Propagator:feedback(feedback)
   if feedback then
      if torch.type(feedback) == 'table' then
         --if list, make visitor_chain
         feedback = dp.CompositeFeedback{feedbacks=feedback}
      end
      self._feedback = feedback
      return
   end
   return self._feedback
end

function Propagator:loss(loss)
   if loss then
      assert(torch.isTypeOf(loss, 'nn.Criterion'), "Expecting nn.Criterion instance")
      self._loss = loss
      return
   end
   return self._loss
end

function Propagator:includeTarget(mode)
   -- forward propagates {input, target} instead of just input
   self._include_target = (mode == nil) and true or mode
end

function Propagator:verbose(verbose)
   self._verbose = (verbose == nil) and true or verbose
   if self._feedback then
      self._feedback:verbose(self._verbose)
   end
   if self._observer then
      self._observer:verbose(self._verbose)
   end
end

function Propagator:silent()
   self:verbose(false)
end

function Propagator:type(new_type)
   self.log.info('\t\t reset type as ', new_type)
   if self._loss then
       self._loss:type(new_type)
   end
   if self._feedback then
       self._feedback:type(new_type)
   end
   self.tensorType = new_type
end

function Propagator:float()
   return self:type('torch.FloatTensor')
end

function Propagator:double()
   return self:type('torch.DoubleTensor')
end

function Propagator:cuda()
   return self:type('torch.CudaTensor')
end

function Propagator:int()
   return self:type('torch.IntTensor')
end

function Propagator:long()
   return self:type('torch.LongTensor')
end
