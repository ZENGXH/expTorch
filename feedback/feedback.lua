------------------------------------------------------------------------
--[[ Feedback ]]--
-- Strategy
-- strategies for processing predictions and targets. 
-- Unlike observers, feedback strategies generate reports.
-- Like observers they may also publish/subscribe to mediator channels.
-- When serialized with the model, they may also be unserialized to
-- generate graphical reports (see Confusion).
------------------------------------------------------------------------
local Feedback = torch.class("dp.Feedback")
Feedback.isFeedback = true

function Feedback:__init(config)
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   local args = {}
   dp.helper.unpack_config(args,{config},
      'Feedback', 
      'strategies for processing predictions and targets.',
      {arg='verbose', type='boolean', default=true,
       help='provide verbose outputs every epoch'},
      {arg='selected_output', type='number', default=0,
       help='works for output as table, report on only part of the ourput, if sero then no selection'}
      {arg='name', type='string', req=true,
       help='used to identify report'}
   )
   self.log = loadfile(paths.concat(dp.DPRNN_DIR, 'utils', 'log.lua'))()
   self.log.SetLoggerName(args.name)

   self.selected_output = args.selected_output 
   self._name = args.name
   self._verbose = args.verbose
   self._n_sample = 0
end

function Feedback:setup(config)
   assert(type(config) == 'table', "Setup requires key-value arguments")
   local args = {}
   dp.helper.unpack_config(args,{config},
      'Feedback:setup', 
      'setup the Feedback for mediation and such',
      {arg='mediator', type='dp.Mediator', 
       help='used for inter-object communication. defaults to dp.Mediator()'},
      {arg='propagator', type='dp.Propagator'}
   )
   self._mediator = args.mediator
   self._propagator = args.propagator
   if self._name then
      self._id = args.propagator:id():create(self._name)
   end
   self._name = nil
end

function Feedback:id()
   return self._id
end

function Feedback:name()
   return self._id and self._id:name() or self._name
end

function Feedback:savePath()
   return self:id():toPath()
end

--accumulates information from the batch
function Feedback:add(batch, output, report)
   assert(torch.isTypeOf(batch, 'dp.Batch'), "First argument should be dp.Batch")
   self._n_sample = self._n_sample + batch:nSample()
   if self.selected_output ~= 0 and torch.isTypeOf(output, 'table') then
       self.log.trace('selected_output is used')
       dp.helper.Assertlet(self.selected_output, #output, 'selected_output too large')
       self:_add(batch, output[self.selected_output], report)
    else
        self:_add(batch, output, report)
    end
end

function Feedback:_add(batch, output, report)
end

function Feedback:report()
   return {}
end

function Feedback:reset()
   self._n_sample = 0
   self:_reset()
end

function Feedback:_reset()
end

function Feedback:verbose(verbose)
   self._verbose = (verbose == nil) and true or verbose
end

function Feedback:silent()
   self:verbose(false)
end

function Feedback:nSample()
   return self._n_sample or 0
end
