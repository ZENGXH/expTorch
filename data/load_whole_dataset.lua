
local LoadWholeDataSet, parent = torch.class("dp.LoadWholeDataSet", "dp.DataSet")
-- do not support multi-thread, load all data at once
function LoadWholeDataSet:__init(config)
    parent.__init(self, config)
    self._can_whole_data_loaded = true
end

function LoadWholeDataSet:FillBatchWithSize(batch, batch_size)
    assert(batch.isBatch)
    batch:SetView('input', self:inputs():sub(1, batch_size))
    if self:targets() then
        batch:SetView('target', self:targets():sub(1, batch_size))
    end
    return batch
end

-------------------------------------------------------------------
-- for loadAllOnce dataSet, when calling FillBatchWithSub, just sub 
-- for existing data save in self:GetView('target') is enough
--
function LoadWholeDataSet:FillBatchWithSub(batch, start, stop)
    assert(batch.isBatch, "Expecting dp.Batch at arg 1")
    assert(batch:IsFilled())
    batch:SetView('input', 
            self:GetView('input'):CreateSubViewWithSub(start, stop))
    if self:targets() then
        batch:SetView('target', 
            self:GetView('target'):CreateSubViewWithSub(start, stop))
    end
    return batch  
end

function LoadWholeDataSet:FillBatchWithIndex(batch, indices)
    assert(batch.isBatch, "Expecting dp.Batch at arg 1")
    assert(batch:IsFilled())
    batch:SetView('input', 
            self:GetView('input'):CreateBatchWithIndex(indices))
    if self:targets() then
        batch:SetView('target', 
            self:GetView('target'):CreateBatchWithIndex(indices))
    end
    return batch  
end


