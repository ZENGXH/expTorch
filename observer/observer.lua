------------------------------------------------------------------------
--[[ Observer ]]--
-- An object that is called when events occur.
-- Based on the Subject-Observer design pattern. 
-- Uses a mediator to publish/subscribe to channels.
-- Observers cannot publish reports (for now). 

-- The reason for this is 
-- that a report is required of observers to do their job, thus making
-- it impossible for them to participate to report generation.
-- The only possibility would be to allow for Observers to modify the 
-- received report, but then the ordering of these modifications would
-- be undefined, unless they make use of Mediator priorities.
------------------------------------------------------------------------
local Observer, parent = torch.class("dp.Observer", "dp.Module")
Observer.isObserver = true

-----------------------------------------------------------------------
-- _init a Observer
-- @param channels: string | table eg. "doneEpoch"
-- @param callbacks: string | table, f not given, same with channels
-----------------------------------------------------------------------
function Observer:__init(config, ...)
   local args = {}
   if type(config) ~= 'table' then
       local args = {...}
       local config_ = {}
       config_.channels = config
       config_.callbacks = args[1]
       config = config_ 
   end
   config = config or {}
   print('unpack: ', config)
   dp.helper.unpack_config(args, {config}, 'observer',
      'Observer for Experiment',
      {arg='channels', type="string | table", 
      help='channels'},
      {arg='callbacks', type="string | table", 
      help='callbacks'},
      {arg='name', type='string', req=true, help='name of Observer'}
   )
   if type(channels) == 'string' then
      channels = {channels}
   end
   if type(callbacks) == 'string' then
      callbacks = {callbacks}
   end
   self._channels = channels or {}
   self._callbacks = callbacks or channels
   args.name = args.name or 'obs'
   parent.__init(self, args)
   -- self.log:= dp.log() -- loadfile(paths.concat(dp.DPRNN_DIR, 'utils', 'log.lua'))()
   -- self.log:SetLoggerName('Observer')
end

-----------------------------------------------------------------------
-- subscribe the channel and callback 
function Observer:subscribe(channelNamespace, callback)
   -- channelNamespace, self as the subscribes, func_name: callback
   self._mediator:subscribe(channelNamespace, self, callback or channel)
end

-- @param subject: Propagator?
-- should be reimplemented to validate subject
function Observer:setSubject(subject)
   --assert subject.isSubjectType
   self._subject = subject
end

----------------------------------------------------------------------
--An observer is setup with 
--a mediator 
--a subject. e,g, the `Experiment` where setup the observer
--The subject is usually the object from which the observer is setup.
----------------------------------------------------------------------
function Observer:setup(config)
   assert(type(config) == 'table', "Setup requires key-value arguments")
   local args, mediator, subject = xlua.unpack(
      {config},
      'Observer:setup', nil,
      {arg='mediator', type='dp.Mediator', req=true},
      {arg='subject', type='dp.Experiment | dp.Propagator | ...',
      help='object being observed.'}
   )
   assert(mediator.isMediator)
   self._mediator = mediator
   self:setSubject(subject)
   for i=1,#self._channels do
      self:subscribe(self._channels[i], self._callbacks[i])
   end
end

function Observer:report()
   error"NotSupported : observers don't generate reports"
end

function Observer:verbose(verbose)
   self._verbose = (verbose == nil) and true or verbose
end

function Observer:silent()
   self:verbose(false)
end

---------------------------------------------------------------
-- look foe the attribute with `name` in the reportm 
-- may `publish`/ notify to _mediator
function Observer:doneEpoch(report, ...)
    error('abstrace method')
end

function Observer:doneBatch(report, ...)
    self.log:trace('not implement')
end
