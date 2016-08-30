local log = loadfile(paths.concat(dp.DPRNN_DIR, 'utils', 'log.lua'))()
local helper = torch.class("dp.helper")

function helper:__init()
end

function helper.unpack(obj, args, funcname, description, ...)
    helper.unpack_config(obj, args, funcname, description, ...)
end

function helper.unpack_config(obj, args, funcname, description, ...)
    xlua.unpack_class(object, args, funcname, description, ...)
    local args = xlua.unpack(args, funcname, description, ...)
      for k,v in pairs(dargs) do
            if type(k) ~= 'number' then
                object[k] = v
            end
      end
end

function helper.WeightInitUniform(net, v)
    local v = v or 0.01
    local p, g = net:getParameters()
    p:uniform(-v, v)
    return net
end

function helper.GetSize(a)
    return a:size():totable()
end
  
function helper.PrintSize(a)
    if a:dim() == 1 then
        return string.format('dim(1) %d', table.unpack(helper.GetSize(a)))
    elseif a:dim() == 2 then
        return string.format('dim(2) %d, %d', table.unpack(helper.GetSize(a)))
    elseif a:dim() == 3 then
        return string.format('dim(3) %d, %d, %d', table.unpack(helper.GetSize(a)))
    elseif a:dim() == 4 then
        return string.format('dim(4) %d, %d, %d, %d', table.unpack(helper.GetSize(a)))
    elseif a:dim() == 5 then
        return string.format('dim(5) %d, %d, %d, %d, %d', table.unpack(helper.GetSize(a)))
    elseif a:dim() == 6 then
        return string.format('dim(6) %d, %d, %d, %d, %d', table.unpack(helper.GetSize(a)))
    end
end

function helper.MakeContiguous(i)
    if not i:isContiguous() then
        local i_cont = torch.Tensor():resizeAs(i):copy(i)
        i = i_cont
    end
    return i
end

function helper.CheckFileExist(filepath)
    local file = io.open(filepath, 'r')
    if file then
        file:close()
        return true
    else
        log.fatal('file '..filepath..' not exist ')
        return false
    end
end

function helper.CheckPathExist(dir)
    if not dir then
        return log.fatal('dir is empty')
    end

    print ('CheckPathExist '..dir)
    if not path.exists(dir) then
        log.fatal('path '..dir..' not exist')
        return false
    else
        return true
    end
end

function helper.LoadBinFile(filepath, datatype)
    local datatype = datatype or 'float'
    local file = torch.DiskFile(filepath, "r"):binary()
    local shape = {}
    for i = 1, 4 do
        local a = file:readInt(1)
        shape[i] = a[1]
    end
    local total = shape[1] * shape[2] * shape[3] * shape[4]
    
    local data
    if datatype == 'float' then
        data = file:readFloat(total)
    else 
        logger.fatal('not implement')
    end
    -- print('data', data)
    data = torch.FloatTensor(data)
    -- print('size of data: ', shape[1], shape[2], shape[3], shape[4])
    
    file:close()
    return data -- return 1024 feature if seg5. 101 feature if prob
end

function helper.GetBugLocation()
    local info = debug.getinfo(3, "Sl")
    local lineinfo = "^^^ "..info.short_src .. ":" .. info.currentline
    return lineinfo
end

function helper.Asserteq(a, b, msg)
    local msg = msg or '.'
    assert(a == b, string.format('%s > check not equal fail: %d == %d  %s', helper.GetBugLocation(), a, b, msg))
end

function helper.Assertlet(a, b, msg)
    local msg = msg or '.'
    assert(a <= b, string.format('%s > check not equal fail: %d <= %d  %s', helper.GetBugLocation(), a, b, msg))
end

function helper.Assertlt(a, b, msg)
    local msg = msg or '.'
    assert(a < b, string.format('%s > check not equal fail: %d < %d  %s', helper.GetBugLocation(), a, b, msg))
end

function helper.Assertnq(a, b, msg)
    local msg = msg or '.'
    assert(a ~= b, string.format('%s > check not equal fail: %d ~= %d  %s', helper.GetBugLocation(), a, b, msg))
end

function helper.TrackMemory(free, msg)
   collectgarbage()
    require 'cutorch'
    local msg = msg or '..'
    local old_free = free or 0
    local freeMemory, totalMemory = cutorch.getMemoryUsage(cutorch.getDevice())
    log.info(string.format('>> %s TrackMemory >> freeMemory from %.3f to %.3f %s', helper.GetBugLocation, old_free/(1024 * 1024), freeMemory/(1024*1024), msg))
    return freeMemory
end

return helper
