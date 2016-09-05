-- WORK IN PROGRESS : DO NOT USE
-- convert for use with dp.Loss instread of nn.Criteria.
-- make non-composite
------------------------------------------------------------------------
--[[ Criteria ]]--
-- Feedback
-- Adapter that feeds back and accumulates the error of one or many
-- nn.Criterion. Each supplied nn.Criterion requires a name for 
-- reporting purposes. Default name is typename minus module name(s)
------------------------------------------------------------------------
local Criteria, parent = torch.class("dp.Criteria", "dp.Feedback")
Criteria.isCriteria = true

function Criteria:__init(config)
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   local args = {}
   criteria, name, typename_pattern = 
   dp.helper.unpack_config(args, {config},
      'Criteria', nil,
      {arg='output_to_report', type='boolean', default=false,
       helper='write result to the report or not'},
      {arg='criteria', type='nn.Criterion | table', req=true,
       help='list of criteria to monitor'},
      {arg='name', type='string', default='criteria'},
      {arg='output_module', type='table', default={nn.Identity()}, help='output module apply to output before pass the criteriion'},
      {arg='typename_pattern', type='string', 
       help='require criteria to have a torch.typename that ' ..  'matches this pattern', default="^nn[.]%a*Criterion$"}
   )
   -- assert
   self.output_to_report = args.output_to_report
   local output_module = args.output_module
   local criteria = args.criteria
   local name = args.name
   local typename_pattern = args.typename_pattern
   assert(torch.type(output_module) == 'table', 'output_module must in tabke get '..torch.type(output_module))
   assert(torch.isTypeOf(output_module[1], 'nn.Module'), 'output_module[1] is a '..torch.type(output_module[1]))
   config.name = name
   parent.__init(self, config)
   self._criteria = {}
   self._output_module = {}
   self._name = name
   if torch.typename(criteria) then
      criteria = {criteria}
   end
   for k,v in pairs(criteria) do
      -- non-list items only
      -- if type(k) ~= 'number' then
         self.log.trace('insert criteria: ', v, self._output_module[k])
         self._criteria[k] = v
         self._output_module[k] = args.output_module[k] or nn.Identity()
      -- end
   end
   
   for i, v in ipairs(criteria) do
      -- for listed criteria, default name is derived from typename
      self._criteria[i] = v
   end
   
   self._errors = {}
   self:reset()
   self.log.tracefrom('Criteria setup done')
end

function Criteria:_reset()
   -- reset error sums to zero
   self.log.info('resetting: ', self._criteria)
   for k, v in pairs(self._criteria) do
      self.log.info('resetting: ', v)
      self._errors[k] = 0
   end
end

function Criteria:_add(batch, output,  report)             
   local current_error
   for k, v in pairs(self._criteria) do
      -- current_error = v:forward(output.act:data(), batch:targets():data())
      self.log.trace('forward ', v)      
      local new_output = self._output_module[k]:forward(output)
      current_error = v:forward(new_output, batch:targets():input())
      self._errors[k] =  (
                              ( self._n_sample * self._errors[k] ) 
                              + 
                              ( batch:nSample() * current_error )
                         ) 
                         / 
                         self._n_sample + batch:nSample()
      --TODO gather statistics on backward outputGradients?
   end
   if self.output_to_report == true then
      report.batch_error = current_error 
   end
end

function Criteria:report()
   return { 
      [self:name()] = self._errors,
      n_sample = self._n_sample
   }
end
