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
log.level = "trace"

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


local _tostring = tostring

local tostring = function(...)
  local t = {}
  for i = 1, select('#', ...) do
    local x = select(i, ...)
    if type(x) == "number" then
      x = round(x, .01)
    end
    t[#t + 1] = _tostring(x)
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
    local info_up = debug.getinfo(3, "sl")
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
        msg = msg..'-fr-'..lineinfo_up
    end
    print(string.format("%s[%-6s%s]%s %s: %s",
                        log.usecolor and x.color or "",
                        nameupper,
                        os.date("%H:%M:%S"),
                        log.usecolor and "\27[0m" or "",
                        lineinfo,
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
