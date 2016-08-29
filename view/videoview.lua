------------------------------------------------------------------------
--[[ VideoView ]]-- 
-- A View holding a tensor of videoSequence.
------------------------------------------------------------------------
local VideoView, parent = torch.class("dp.VideoView", "dp.DataView")
VideoView.isVideoView = true
function VideoView:__init(name)
    local name = name or 'VideoView'
    parent.__init(self, name)
end

-- batch x height x width x channels/colors x time
function VideoView:bhwct()
    if self._view == 'bhwct' then
        return nn.Identity()
    end
    return self:transpose('bhwct')
end

function VideoView:tbchw()
    if self._view == 'bhwct' then
        return nn.Identity()
    end
    return self:transpose('bhwct')
end

function VideoView:hasAxis(axis_char, view)
    view = view or self._view
    local axis_pos = view:find(axis_char)
    if not axis_pos then
        return false
    else
        return true
    end
end


-- View used by SpacialConvolutionCUDA
function VideoView:chwb()
    if self._view == 'chwb' then
        return nn.Identity()
    end
    if #self._dim == 4 and not self:hasAxis('t') then
        return self:transpose('chwb')
    else
        --[[ if #self._dim == 5 then
        local t_pos = self:findAxis('t')
        if self._input:size(t_pos) == 1 then
            local modula = nn.Sequential()
            modula:add(nn.Select(t_pos, 1))
            modula:add(self:transpose('chwb'))
        end
    end]]--
        error('not Implemented for '..self._view..' to chwb')
    end
end


-- View used by SpacialConvolution
function VideoView:bchw()
    if self._view == 'bchw' then
        return nn.Identity()
    end
    return self:transpose('bchw')
end

function VideoView:tbchw()

end

-- for temporal convolution
function VideoView:btc()
    if #self._view ~= 3 then 
        error("Cannot convert view '"..self._view.."' to 'btc'")
    end
    if self._view == 'btc' then
        return nn.Identity()
    end
    return self:transpose('btc')
end
