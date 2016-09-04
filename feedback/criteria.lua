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
   local args, criteria, name, typename_pattern = xlua.unpack(
      {config},
      'Criteria', nil,
      {arg='criteria', type='nn.Criterion | table', req=true,
       help='list of criteria to monitor'},
      {arg='name', type='string', default='criteria'},
      {arg='typename_pattern', type='string', 
       help='require criteria to have a torch.typename that ' ..
       'matches this pattern', default="^nn[.]%a*Criterion$"}
   )
   config.name = name
   parent.__init(self, config)
   self._criteria = {}
   self._name = name
   if torch.typename(criteria) then
      criteria = {criteria}
   end
   
   for k,v in pairs(criteria) do
      -- non-list items only
      if type(k) ~= 'number' then
         self._criteria[k] = v
      end
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

function Criteria:_add(batch, output, carry, report)             
   local current_error
   local new_output
   if torch.isTypeOf(output, 'table') and self.selected_output then
        new_output = output[1]
   else
       new_output = output
   end
    assert(torch.isTensor(new_output))
   for k, v in pairs(self._criteria) do
      -- current_error = v:forward(new_output.act:data(), batch:targets():data())
      
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
end

function Criteria:report()
   return { 
      [self:name()] = self._errors,
      n_sample = self._n_sample
   }
end
