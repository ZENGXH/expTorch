------------------------------------------------------------------------
--[[ BaseSet ]]--
-- Base class inherited by DataSet and Batch.
-- @Method:
-- 1. get and set the members by [which_set][attribute]
--
-- 2. set IO_preprocess to inputsVIew and targetsVIew at the same time
-- both dataset and batch have inputView and targetView, and Preprocesses
-- 
------------------------------------------------------------------------
local BaseSet = torch.class("dp.BaseSet")
BaseSet.isBaseSet = true

------------------------------------------------------------------------
-- create a new BaseSet
-- Use to manage inputsDataView & targetDataView for train/valid/test dataset
-- can be imageDataSet ot VideoDataSet, work with classDataSet(GT)
--
-- @param config: table: include
--  which_set: string = 'train'
--  [inputs: View] -- DEPRECATED
--  [targets: View] -- DEPRECATED
------------------------------------------------------------------------
function BaseSet:__init(config)
   self._has_input_view = false
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   local args = {}
   -- name, which_set, inputs, targets
   dp.helper.unpack_config(args, {config},
      'BaseSet', 
      'Base class inherited by DataSet and Batch.',
      {arg='name', type='string', default=' ', help='name of the set'},
      {arg='which_set', type='string', req=true,
      help='"train", "valid" or "test" set'},
      {arg='inputs', type='dp.View | table of dp.Views', 
       help='Sample inputs to a model. These can be Views or '..
       'a table of Views (in which case these are converted '..
       'to a ListView'},
      {arg='targets', type='dp.View | table of dp.Views', 
       help='Sample targets to a model. These can be Views or '..
       'a table of Views (in which case these are converted '..
       'to a ListView. The indices of examples must be '..
       'in both inputs and targets must be aligned.'}
   )
   self.log = loadfile(paths.concat(dp.DPRNN_DIR, 'utils', 'log.lua'))()
   self.log.SetLoggerName(args.name)
   self:whichSet(args.which_set)
   self._has_target_view = false
   if args.inputs then 
       self.log.fatal('set inputs explictly please')
       self:SetView('inputs', args.inputs)
   end
   if args.targets then 
       self.log.fatal('set targets explictly please')
       self:SetView('targets', args.targets)
   end
end

function BaseSet:whichSet(which_set)
   if which_set then
      self._which_set = which_set
   end
   return self._which_set
end

function BaseSet:isTrain()
   return (self._which_set == 'train')
end

-- Returns the number of samples in the BaseSet.
function BaseSet:nSample()
   if self._n_sample then
      return self._n_sample
   elseif self._inputs then
      return self._inputs:nSample()
   elseif self._targets then
      return self._targets:nSample()
   else
      return 0
   end
end
----------------------------------------------------
-- get/set input/target dp.View
-- @param attribute: string, input or target
-- 
-----------------------------------------------------
function BaseSet:GetView(attribute)
   assert(attribute, 'attribute nil')
   if attribute == 'inputs' or attribute == 'input' then
      if not self._has_input_view then 
         self.log.tracefrom('\t get nil') 
         return nil
      else
         return self._inputs
      end
   elseif attribute == 'target' or attribute == 'targets' then
      if not self._has_target_view then 
         self.log.tracefrom('\t get nil') 
         return nil
      else
         return self._targets
      end
  else
     error('invalid attribute: ', attribute)
  end
end

function BaseSet:SetView(attribute, dataview)
   assert(dataview.isView, "Error : invalid inputs. Expecting type dp.View")
   if attribute == 'inputs' or attribute == 'input' then
      self._inputs = dataview 
      self._has_input_view = true
      return self._inputs
   elseif attribute == 'target' or attribute == 'targets' then
      self._targets = dataview
      self._has_target_view = true
      return self._targets
  else
     error('invalid attribute')
  end
end

function BaseSet:IsFilled()
   return self._has_input_view
end

----------------------------------------------------
-- Preprocesses are applied to Views
-- @param config
---------------------------------------------------
function BaseSet:SetPreprocess(config)
   config = config or {}
   assert(torch.type(config) == 'table' and not config[1], 
      "Constructor requires key-value arguments")
   local args, input_preprocess, target_preprocess, can_fit
      = xlua.unpack(
         {config},
         'BaseSet:preprocess',
         'Preprocesses the BaseSet.',
         {arg='input_preprocess', type='dp.Preprocess', 
          help='Preprocess applied to the input View of the BaseSet'},
         {arg='target_preprocess', type='dp.Preprocess',
          help='Preprocess applied to the target View of the BaseSet'},
         {arg='can_fit', type='boolean',
          help='Allows measuring of statistics on the View ' .. 
          'of BaseSet to initialize the preprocess. Should normally ' .. 
          'only be done on the training set. Default is to fit the ' ..
          'training set.'} --? measuring of statistics?
   )
   assert(input_preprocess or target_preprocess, 
      "Error: no preprocess (neither input nor target) provided)")
   if can_fit == nil then
      can_fit = self:isTrain()
   end
   --TODO support multi-input/target preprocessing
   if input_preprocess and input_preprocess.isPreprocess then
      input_preprocess:apply(self._inputs, can_fit)
   end
   if target_preprocess and target_preprocess.isPreprocess then
      target_preprocess:apply(self._targets, can_fit)
   end
end

---------------------------------------------------------------
-- become DEPRECATED
function BaseSet:preprocess(config)
    self.log.fatal('DEPRECATED cann SetPreprocess instead')
    return self:SetPreprocess(config)
end

function BaseSet:inputs(inputs)
   if inputs then
      self.log.tracefrom('set dataView: inputs')
      return self:SetView('input', inputs)
   else
      self.log.tracefrom('request dataView: inputs')
      return self:GetView('input')
   end
end

-- get/set target dp.View
function BaseSet:targets(targets)
   if targets then
      self.log.tracefrom('set dataView: targets')
      return self:SetView('target', targets)
   end
   self.log.tracefrom('request dataView: targets')
   return self:GetView('target')
end
-- BEGIN DEPRECATED (June 13, 2015)
function BaseSet:setInputs(inputs)
   assert(inputs.isView, 
      "Error : invalid inputs. Expecting type dp.View")
   self._inputs = inputs
end

function BaseSet:setTargets(targets)
   assert(targets.isView,
   "Error : invalid targets. Expecting type dp.View")
   self._targets = targets
end

function BaseSet:setWhichSet(which_set)
   self._which_set = which_set
end
-- END DEPRECATED

-- vim:ts=3 ss=3 sw=3 expandtab
