--
-- log.lua
--
-- Copyright (c) 2016 rxi
-- Contributions by
--  ZENG Xiaohui xzengaf@ust.hk  
-- This library is free software; you can redistribute it and/or modify it
-- under the terms of the MIT license. See LICENSE for details.
--

local log = { _version = "0.1.0" }

log.usecolor = true
log.outfile = log.outfile or paths.concat(os.getenv('PWD'), 'log')
log.level = "info"
log.name = " " 

function log.SetLoggerName(name)
    --print('set log name as ', name)
    log.name = name
end

local modes = {
    { name = "trace", color = "\27[34m", },
    { name = "tracefrom", color = "\27[34m"},
    { name = "debug", color = "\27[36m", },
    { name = "write", color = "\27[32m", },
    { name = "info",  color = "\27[32m", },
    { name = "warn",  color = "\27[33m", },
    { name = "error", color = "\27[31m", },
    { name = "fatal", color = "\27[35m", },
}


local levels = {}
for i, v in ipairs(modes) do
    levels[v.name] = i
end


local round = function(x, increment)
    increment = increment or 1
    x = x / increment
    return (x > 0 and math.floor(x + .5) or math.ceil(x - .5)) * increment
end



local tostring2 = function(obj, ...)
    local msg = {}
    if type(obj) == 'table' then
        local mt = getmetatable(obj)
        if mt and mt.__tostring__ then
            msg[#msg + 1] = mt.__tostring__(obj)
        else
            local tos = tostring(obj)
            local obj_w_usage = false
            if tos and not string.find(tos, 'table: ') then
                if obj.usage and type(obj.usage) == 'string' then
                    msg[#msg+1] = obj.usage
                    msg[#msg+1] = '\n\nFIELDS:\n'
                    obj_w_usage = true
                else
                    msg[#msg+1] = tos .. ':\n'
                end
            end
            msg[#msg+1] = '{'
            local tab = ''
            local idx = 1
            for k, v in pairs(obj) do
                if idx > 1 then msg[#msg + 1] = ',\t' end
                if type(v) == 'userdata' then
                    msg[#msg + 1] = tab..'['.. k ..']'..'= <userdata>'
                else
                    local tostr = tostring(v):gsub('\n','\\n')
                    if #tostr>40 then
                        local tostrshort = tostr:sub(1,40) .. sys.COLORS.none
                        msg[#msg + 1] = tab..'['..tostring(k) ..']'..' = '..tostrshort..' ... '
                    else
                        msg[#msg + 1] = tab..'['.. tostring(k)..']'..' = ' .. tostr
                    end
                end
                tab = ' '
                idx = idx + 1
            end
            msg[#msg + 1] = '}'
            if obj_w_usage then msg[#msg + 1 ] = '' end
        end
    else
        msg[#msg + 1] = tostring(obj)
    end
    if select('#',...) > 0 then
        msg[#msg + 1] = '..'
    else
        msg[#msg + 1] = ' '
    end
    return table.concat(msg, " ")
end
---------
local _tostring = tostring

local tostring = function(...)
    local t = {}
    for i = 1, select('#', ...) do
        local x = select(i, ...)
        if type(x) == "number" then
            x = round(x, .001)
        end
        t[#t + 1] = tostring2(x)
    end
    return table.concat(t, " ")
end
for i, x in ipairs(modes) do

    local nameupper = x.name:upper()
    log[x.name] = function(...)

        -- Return early if we're below the log level
        if i < levels[log.level] then
            return
        end
        -- if torch.isinstance(..., tensor) then

        -- end
        local msg = tostring(...)
        local info = debug.getinfo(2, "Sl")
        local lineinfo = info.short_src .. ":" .. info.currentline
        local info_up = debug.getinfo(3, "Sl")
        local lineinfo_up = "("..info_up.short_src .. ":" .. info_up.currentline..")"

        -- Output to console
        if log.start_with then
            msg = log.start_with..msg
        end
        if x.name == "write" then
            local fp = io.open("out", "a")
            local str = string.format("[%-6s%s] %s: %s\n",
            nameupper, os.date(), lineinfo, msg)
            fp:write(str)
            fp:close()
            return 
        end   
        if x.name == "tracefrom" then
            msg = msg..'---'..lineinfo_up
        end

        print(string.format("%s[%-6s%s]%s %s:{%s} %s",
        log.usecolor and x.color or "",
        nameupper,
        os.date("%H:%M:%S"),
        log.usecolor and "\27[0m" or "",
        lineinfo,
        log.name,
        msg))


        -- Output to log file
        if log.outfile then
            local fp = io.open(log.outfile, "a")
            local str = string.format("[%-6s%s] %s: %s\n",
            nameupper, os.date(), lineinfo, msg)
            fp:write(str)
            fp:close()
        end

    end
end


return log
