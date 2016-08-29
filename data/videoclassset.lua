local VideoClassSet, parent = torch.class("dp.ImageClassSet", "dp.DataSet")
local log = dp.log
VideoClassSet._input_shape = 'bchw'
VideoClassSet._output_shape = 'b'

function VideoClassSet:__init(config)
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   local args, data_path, load_size, sample_size, sample_func, which_set,  
      verbose, sort_func, cache_mode, cache_path = xlua.unpack(
      {config},
      'VideoClassSet', 
      'A DataSet for images in a flat folder structure',
      {arg='data_list', type='string', req=true,
       help='dataList contain all the filename, length, label'},

      {arg='data_path', type='table | string', req=true,
       help='one or many paths of directories with images'},

      {arg='load_size', type='table', req=true,
       help='a size to load the images to, initially'},
        
      {arg='sample_size', type='table', req=true
       help='a consistent sample size to resize the feature regardless of the num_frames, can be d,h,w or d . '..
       'Defaults to load_size'},

      {arg='sample_func', type='string | function', default='sampleDefault',
       help='function f(self, dst, path) used to create a sample(s) from '..
       'an image path. Stores them in dst. Strings "sampleDefault", '..
       '"sampleTrain" or "sampleTest" can also be provided as they '..
       'refer to existing functions'},

      {arg='which_set', type='string', default='train',
       help='"train", "valid" or "test" set'},

      {arg='verbose', type='boolean', default=true,
       help='display verbose messages'},

      {arg='sort_func', type='function', 
       help='comparison operator used for sorting class dir to get idx.'
       ..' Defaults to < operator'},

      {arg='cache_mode', type='string', default='writeonce',
       help='writeonce : read from cache if exists, else write to cache. '..
       'overwrite : write to cache, regardless if exists. '..
       'nocache : dont read or write from cache. '..
       'readonly : only read from cache, fail otherwise.'},

      {arg='cache_path', type='string', 
       help='Path to cache. Defaults to [data_path[1]]/cache.th7'}
   )
   -- globals
   gm = require 'graphicsmagick'
   
   -- locals
   self:whichSet(which_set)
   self._load_size = load_size
   assert(self._load_size[1] == 3, "VideoClassSet doesn't yet support greyscaling : load_size")
   self._sample_size = sample_size or self._load_size
   assert(self._sample_size[1] == 3, "VideoClassSet doesn't yet support greyscaling : sample_size")
   self._verbose = verbose   
   self._data_path = type(data_path) == 'string' and {data_path} or data_path
   self._sample_func = sample_func
   -- TODO: assert input argument sample_func is valid
   self._sort_func = sort_func
   self._cache_mode = cache_mode
   self._cache_path = cache_path or paths.concat(self._data_path[1], 'cache.th7')
   
   -- indexing and caching
   assert(_.find({'writeonce','overwrite','nocache','readonly'},cache_mode), 'invalid cache_mode :'..cache_mode)
   local cacheExists = paths.filep(self._cache_path)
   if cache_mode == 'readonly' or (cache_mode == 'writeonce' and cacheExists) then
      if not cacheExists then
         error"'readonly' cache_mode requires an existing cache, none found"
      end
      self:loadIndex()
   else
      self:buildIndex()
      if cache_mode ~= 'nocache' then
         self:saveIndex()
      end
   end
   
   -- buffers
   self._imgBuffers = {}
   
   -- required for multi-threading
   self._config = config 
end

function VideoClassSet:saveIndex()
   local index = {}
   for i, k in ipairs{'_classes','_classIndices','imagePath','imageClass','classList','classListSampleTitle'} do
      index[k] = self[k]
   end
   torch.save(self._cache_path, index)
end

function VideoClassSet:loadIndex()
   local index = torch.load(self._cache_path)
   for k, v in pairs(index) do
      self[k] = v
   end
   self._n_sample = self.imagePath:size(1)
end

function VideoClassSet:buildIndex()
   -- loop over each paths folder, get list of unique class names, 
   -- also store the directory paths per class
   -- {'_videoes','_videoIndices', '_videolabel', 'imagePath','imageClass','classList','classListSampleTitle'} do
   -- local classes = {} -- {'v_aas': true, 'cv_df': true, ...}
   -- local classList = {} -- {'v_applysss', 'v_running', 'v_fdf'}

    local classList = {} 
    local classes = {}
    local classListVideo = {}
    local videoList = {}
    local videoIndices = {}
    local videoLabel = {}
    local videoLength = {}
    local videoPath = {}

    local data_dict = dp.ucf101_helper.ReadGTText2DataDict(
        self.classID_file, 
        self.input_file, 
        classList, classes, classListVideo, classIndices
        videoList, videoIndices, videoLabel, videoLength, videoPath)
    self.classList = classList 
    self._classes = classes
    self._classIndices = classIndices
    self.classListVideo = classListVideo
    self.videoList = videoList
    self.videoIndices = videoIndex
    self.videoLength = videoLength
    self.videoPath = videoPath
    
    local runningIndex = 0
    seld.videoClass = torch.Tensor(#self.videoList):fill(0)
    self.classListVideoIndices = {}
    for index_class in 1, #self._classes do
        local num_video = #self.classListVideo[index_class]
        if num_video == 0 then
            error('Class has zero samples')
        else
            self.classListVideoIndices[index_class] = torch.linspace(
                runningIndex+1, runningIndex+num_video, num_video)
            self.videoClass[{{runningIndex + 1, runningIndex + num_video}}]:fill(index_class)
        end
        runningIndex = runningIndex + num_video
    end
    
    runningIndex = 0
    self.classListFrameIndices = {}
    self,framesClass = torch.Tensor():resize(torch.Tensor(self.videoLength):sum())
    for index_video in 1, #self._videoLists do
        local num_frames = self.videoLength[self.videoList[index_video]]
        assert(num_frames >0 )
        self.classListVideoIndices[index_video] = torch.linspace(
                runningIndex+1, runningIndex+num_frames, num_frames)
        self.framesClass[{{runningIndex + 1, runningIndex + num_frames}}]:fill(self.videoClass[index_video])
        runningIndex = runningIndex + num_frames
    end

    if self._verbose then
      print("found " .. #self._classes .. " classes")
    end
   
   ---------------------------------------------------------------------
   -- find the image path names
   self._n_sample = #self.videoList
   self._n_video = self.videoClass:size(1)
   self._n_frame = self.framesClass:size(1)
   ---------------------------------------------------------------------
   if self._verbose then
      print('Updating classList and videoLabel appropriately')
   end

end

function VideoClassSet:batch(batch_size)
   return dp.Batch{
      which_set=self._which_set,
      inputs=dp.ImageView('bchw', 
        torch.FloatTensor(batch_size, unpack(self._sample_size))),
      targets=dp.ClassView('b', 
        torch.IntTensor(batch_size))
   }
end

-- nSample(), nSample(class)
function VideoClassSet:nSample(class, list)
   local list = list or self.classList
   if not class then
      return self._n_sample
   elseif type(class) == 'string' then
      return list[self._classIndices[class]]:size(1)
   elseif type(class) == 'number' then
      return list[class]:size(1)
   end
end

function VideoClassSet: sub(batch, start, stop)
   if not stop then
      stop = start
      start = batch
      batch = nil
   end
   -- inie a batch nil
   batch = batch or dp.Batch{
        which_set=self:whichSet(), 
        epoch_size=self:nSample()
        }
   -- convert [string] self._sample_func to [function] sampleFunc
   local sampleFunc = self._sample_func
   if torch.type(sampleFunc) == 'string' then
      sampleFunc = self[sampleFunc]
   
   
   local inputTable = {}
   local targetTable = {}
   local i = 1
   for idx=start, stop do
      -- load the sample
      local videopath = ffi.string(torch.data(self.videoPath[idx]))
      local dst = self:getImageBuffer(i)
      dst = sampleFunc(self, dst, videopath) -- pass video path to sample function
      table.insert(inputTable, dst)
      table.insert(targetTable, self.videoLabel[idx])     
      i = i + 1
   end

   local inputView = batch and batch:inputs() or dp.ImageView()
   local targetView = batch and batch:targets() or dp.ClassView()
   local inputTensor = inputView:input() or torch.FloatTensor()
   local targetTensor = targetView:input() or torch.IntTensor()

   self:tableToTensor(inputTable, targetTable, inputTensor, targetTensor)

   -- assert(inputTensor:size(2) == 3)
   inputView:forward('btchw', inputTensor)
   targetView:forward('b', targetTensor)
   targetView:setClasses(self._classes)
   batch:inputs(inputView)
   batch:targets(targetView)  

   return batch
end

function VideoClassSet:index(batch, indices)
   if not indices then
      indices = batch
      batch = nil
   end
   batch = batch or dp.Batch{which_set=self:whichSet(), 
        epoch_size=self:nSample()}

   local sampleFunc = self._sample_func
   if torch.type(sampleFunc) == 'string' then
      sampleFunc = self[sampleFunc]
   end

   local inputTable = {}
   local targetTable = {}
   for i = 1, indices:size(1) do
      idx = indices[i]
      -- load the sample
      local imgpath = ffi.string(torch.data(self.videoPath.imagePath[idx]))
      local dst = self:getImageBuffer(i)
      dst = sampleFunc(self, dst, imgpath)
      table.insert(inputTable, dst)
      table.insert(targetTable, self.imageClass[idx])
   end

   local inputView = batch and batch:inputs() or dp.VideoView()
   local targetView = batch and batch:targets() or dp.ClassView()
   local inputTensor = inputView:input() or torch.FloatTensor()
   local targetTensor = targetView:input() or torch.IntTensor()

   self:tableToTensor(inputTable, targetTable, inputTensor, targetTensor)

   -- assert(inputTensor:size(2) == 3)
   inputView:forward('bchw', inputTensor)
   targetView:forward('b', targetTensor)
   targetView:setClasses(self._classes)
   batch:inputs(inputView)
   batch:targets(targetView)
   return batch
end

-- converts a table of samples (and corresponding labels) to tensors
function VideoClassSet:tableToTensor(inputTable, targetTable, inputTensor, targetTensor)
   -- inputTensor possible shape: (t, c, h, w) or (t, c) or (1, c, h, w) or (1, c) or (c) or (c, h, w)
   inputTensor = inputTensor or torch.FloatTensor()
   targetTensor = targetTensor or torch.IntTensor()
   local n = #targetTable -- batchSize

   local framesPerDraw = (inputTable[1]:dim() == 1 or 3) and 1 or inputTable[1]:size(1)
   inputTensor:resize(n, framesPerDraw, unpack(self._sample_size))
   targetTensor:resize(n)
   
   for i = 1, n do
      inputTensor[i]:copy(inputTable[i])
      targetTensor[i]:fill(targetTable[i])
   end
   return inputTensor, targetTensor
end

function VideoClassSet:loadImage(path)
   -- https://github.com/clementfarabet/graphicsmagick#gmimage
   local lW, lH = self._load_size[3], self._load_size[2]
   -- load image with size hints
   local input = gm.Image():load(path, self._load_size[3], self._load_size[2])
   -- resize by imposing the smallest dimension (while keeping aspect ratio)
   local iW, iH = input:size()
   if iW/iH < lW/lH then
      input:size(nil, lW)
   else
      input:size(nil, lH)
   end
   return input
end

-- return a table contain all frames of the video 
function VideoClassSet:loadVideo(path)
    local concat = torch.load(path)
    local data = concat.data
    return data
end

function VideoClassSet:loadVideoWithChecking(path, length, title, label)
    local concat = torch.load(path)
    assert(length == concat['length'])
    -- notices that length ~= #concat.data, need to consider the reduced length
    assert(title == concat['title'])
    assert(label == concat['label'])
    return concat.data
end

function VideoClassSet:getImageBuffer(i)
   self._imgBuffers[i] = self._imgBuffers[i] or torch.FloatTensor()
   return self._imgBuffers[i]
end

-- Sample a class uniformly, and then uniformly samples example from class.
-- This keeps the class distribution balanced.
-- sampleFunc is a function that generates one or many samples
-- from one image. e.g. sampleDefault, sampleTrain, sampleTest.
function VideoClassSet:sample(batch, nSample, sampleFunc)
   if (not batch) or (not sampleFunc) then 
      if torch.type(batch) == 'number' then
         sampleFunc = nSample
         nSample = batch
         batch = nil
      end
      batch = batch or dp.Batch{which_set=self:whichSet(), 
            epoch_size=self:nSample()}   
   end
   
   sampleFunc = sampleFunc or self._sample_func
   if torch.type(sampleFunc) == 'string' then
      sampleFunc = self[sampleFunc]
   end

   nSample = nSample or 1
   local inputTable = {}
   local targetTable = {}   
   for i=1, nSample do
      -- sample class(label)
      local class = torch.random(1, #self._classes)

      -- sample video from class
      local index = torch.random(1, #self.classListSampleTitle[class])
      
      local videoPath = ffi.string(torch.data(self.videoPath[self.classListSampleTitle[class][index]]))
      local dst = self:getImageBuffer(i)
      dst = sampleFunc(self, dst, videoPath)
      table.insert(inputTable, dst)
      table.insert(targetTable, class)  
   end
   
   local inputView = batch and batch:inputs() or dp.VideoView()
   local targetView = batch and batch:targets() or dp.ClassView()
   local inputTensor = inputView:input() or torch.FloatTensor()
   local targetTensor = targetView:input() or torch.IntTensor()
   
   self:tableToTensor(inputTable, targetTable, inputTensor, targetTensor)
   
   -- assert(inputTensor:size(2) == 3)
   inputView:forward('btchw', inputTensor)
   targetView:forward('b', targetTensor)
   targetView:setClasses(self._classes)
   batch:inputs(inputView)
   batch:targets(targetView)  
   
   collectgarbage()
   return batch
end

-- by default, just load the image and return it
function VideoClassSet:sampleDefault(dst, path)
   if not path then
      path = dst
      dst = torch.FloatTensor()
   end
   if not dst then
      dst = torch.FloatTensor()
   end
   -- if load_size[1] == 1, converts to greyscale (y in YUV)
   local out = self:loadVideo(path)
   return out
end

function VideoClassSet:sampleVolume(dst, path)
    local framesPerDraw 
    if path then
        -- if given dst tensor, inder framesPerDraw from the first dimension
        framesPerDraw = dst:size(1) or self.framesPerDraw
        assert(framesPerDraw, 'framesPerDraw must be set!')
    end
    if not path then
        path = dst
        dst = torch.FloatTensor()
    end
    if not dst then
        dst = torch.FloatTensor()
    end
    local out = self:loadVideo(path):view(out:size(1), unpack(self._load_size))
    dst:resize(framesPerDraw, unpack(self._sample_size)):fill(0)
    local num_frames = #out
    local index_start = torch.random(1, num_frames)
    local i = 1
    for index = index_start, math.min(index_start + framesPerDraw - 1, num_frames) do
            dst[i]:copy(out[index])
            i = i + 1
    end
   return dst
end

-- function to load the image, jitter it appropriately (random crops etc.)
function VideoClassSet:sampleTrain(dst, path)
   error('not implement')
   return dst
end

-- function to load the image, do 10 crops (center + 4 corners) and their hflips
-- Works with the TopCrop feedback
function VideoClassSet:sampleTest(dst, path)
    -- TODO: do 25 sampling?
   error('not implement')
   return dst
end

function VideoClassSet:sampleValid(dst, path)
    -- TODO: do 1 sampling
    error('not implement')
end

function VideoClassSet:classes()
   return self._classes
end

------------------------ multithreading --------------------------------

function VideoClassSet:multithread(nThread)
   nThread = nThread or 2
   if not paths.filep(self._cache_path) then
      -- workers will read a serialized index to speed things up
      self:saveIndex()
   end
   
   local mainSeed = os.time()
   local config = self._config
   config.cache_mode = 'readonly'
   config.verbose = self._verbose
   
   local threads = require "threads"
   threads.Threads.serialization('threads.sharedserialize')

   self._threads = threads.Threads(
      nThread,
      -- all function below will be executed in all thread
      function() -- make a separated f1 containing all the definitions 
         require 'dp'
      end,
      function(idx) -- other code in f2
         opt = options -- pass to all donkeys via upvalue
         tid = idx
         local seed = mainSeed + idx
         math.randomseed(seed)
         torch.manualSeed(seed)
         if config.verbose then
            print(string.format('Starting worker thread with id: %d seed: %d', tid, seed))
         end
         dataset = dp.VideoClassSet(config)
         tbatch = dataset:batch(1)
      end
   )
   
   self._send_batches = dp.Queue() -- batches sent from main to threads
   self._recv_batches = dp.Queue() -- batches received in main from threads
   self._buffer_batches = dp.Queue() -- buffered batches
   
   -- public variables
   self.nThread = nThread
   self.isAsync = true
end

-- pull batches from self._recv_batches and push to _buffer_batches
function VideoClassSet:synchronize()
   self._threads:synchronize()
   while not self._recv_batches:empty() do
     self._buffer_batches:put(self._recv_batches:get())
   end
end

-- send request to worker : put request into queue
-- create a batch with batch_size (stop-start+1)
-- put into self._send_batches
-- add thread job, in which :sub is call in data-worker thread
-- main-thread get batch from ._send_batches and put into _recv_batches
function VideoClassSet:subAsyncPut(batch, start, stop, callback)   
   if not batch then
      -- get a batch from _buffer_batches or create a new batch
      -- and pre-filled the [dataView] input abd target
      batch = (not self._buffer_batches:empty()) and self._buffer_batches:get() or self:batch(stop-start+1)
   end

   local input = batch:inputs():input()
   local target = batch:targets():input()
   
   assert(batch:inputs():input() and batch:targets():input())
   
   self._send_batches:put(batch)
   
   self._threads:addjob(
      -- the job callback (runs in data-worker thread)
      function()
         tbatch:inputs():forward('btchw', input)
         tbatch:targets():forward('b', target)
         dataset:sub(tbatch, start, stop)
         return input, target
         -- the callback return one ore many values which will be 
         -- serialized and unserialized as arguments to the endcallback function. 
      end,

      -- the endcallback (runs in the main thread)
      function(input, target)
         local batch = self._send_batches:get()
         -- filling input data
         batch:inputs():forward('btchw', input)
         batch:targets():forward('b', target)
         -- init call batch:setup and do preprocesses 
         callback(batch)
         
         batch:targets():setClasses(self._classes)
         self._recv_batches:put(batch)
      end
   )
end

function VideoClassSet:sampleAsyncPut(batch, nSample, sampleFunc, callback)
   self._iter_mode = self._iter_mode or 'sample'
   if (self._iter_mode ~= 'sample') then
      error'can only use one Sampler per async VideoClassSet (for now)'
   end  
   
   if not batch then
      batch = (not self._buffer_batches:empty()) and self._buffer_batches:get() or self:batch(nSample)
   end
   local input = batch:inputs():input()
   local target = batch:targets():input()
   assert(input and target)
   
   -- transfer the storage pointer over to a thread
   local inputPointer = tonumber(ffi.cast('intptr_t', 
        torch.pointer(input:storage())))
   local targetPointer = tonumber(ffi.cast('intptr_t', 
        torch.pointer(target:storage())))
   input:cdata().storage = nil
   target:cdata().storage = nil
   
   self._send_batches:put(batch)
   
   assert(self._threads:acceptsjob())
   self._threads:addjob(
      -- the job callback (runs in data-worker thread)
      function()
         -- set the transfered storage
         torch.setFloatStorage(input, inputPointer)
         torch.setIntStorage(target, targetPointer)
         tbatch:inputs():forward('bchw', input)
         tbatch:targets():forward('b', target)
         
         dataset:sample(tbatch, nSample, sampleFunc)
         
         -- transfer it back to the main thread
         local istg = tonumber(ffi.cast('intptr_t', 
            torch.pointer(input:storage())))
         local tstg = tonumber(ffi.cast('intptr_t', 
            torch.pointer(target:storage())))
         input:cdata().storage = nil
         target:cdata().storage = nil
         return input, target, istg, tstg
      end,

      -- the endcallback (runs in the main thread)
      function(input, target, istg, tstg)
         local batch = self._send_batches:get()
         torch.setFloatStorage(input, istg)
         torch.setIntStorage(target, tstg)
         batch:inputs():forward('bchw', input)
         batch:targets():forward('b', target)
         
         callback(batch)
         
         batch:targets():setClasses(self._classes)
         self._recv_batches:put(batch)
      end
   )
end

-- recv results from worker : get results from queue
function VideoClassSet:asyncGet()
   -- necessary because Threads:addjob sometimes calls dojob...
   if self._recv_batches:empty() then
      self._threads:dojob()
   end
   
   return self._recv_batches:get()
e
