------------------------------------------------------------------------
--[[ videoclassset ]]--
-- a dataset for video  in a flat folder structure :
-- [data_path]/[vdieo]/[framesname].jpeg  (folder-name is video-name)
-- optimized for extremely large datasets (9000 videos+).
-- video on disk can have different length.
-- video can be reach by reading the whole video ot frames bin
-- currently not modify for frames-level-raw-video-frames input
------------------------------------------------------------------------

local helper = loadfile(paths.concat(dp.DPRNN_DIR, 'utils', 'helper.lua'))()
local VideoClassSet, parent = torch.class("dp.VideoClassSet", "dp.DataSet")
-- batch, timeStep, channels, height, width
VideoClassSet._input_shape = 'btchw' 
VideoClassSet._output_shape = 'b'

function VideoClassSet:__init(config)
    parent.__init(self, config)
    self.log.info('__init VideoClassSet', unpack(config))
    assert(type(config) == 'table', "Constructor requires key-value arguments")
    local args, classID_file,
        video_list, data_path, load_size, sample_size, 
        sample_func, which_set, frames_per_select, 
        verbose, sort_func, cache_mode, cache_path,
        io_helper = xlua.unpack(
    {config},
    'VideoClassSet', 
    'A DataSet for images in a flat folder structure',

    {arg='classID_file', type='string', req=true, 
    help='dataList contain all the filename, length, label'},
    {arg='data_list', type='string', req=true,
    helpi='list for all video'},
    {arg='data_path', type='table | string', req=true,
    help='path to the bin file, the most close upper path including all the data'..
    'should be given'},

    {arg='load_size', type='table', req=true,
    help='a size to load the images to, initially'},
    {arg='sample_size', type='table', req=true,
    help='a consistent sample size to resize the feature'..
    'regardless of the num_frames,can be d,h,w or d, Defaults to load_size'},

    {arg='sample_func', type='string | function', default='sampleDefault',
    help='function f(self, dst, path) used to create a sample(s) from '..
    'an image path. Stores them in dst. Strings "sampleDefault", '..
    '"sampleTrain" or "sampleTest" can also be provided as they '..
    'refer to existing functions'},
    {arg='which_set', type='string', default='train',
    help='"train", "valid" or "test" set'},

    {arg='frames_per_select', type='number', default=1,
    help='num of frames to be select'},
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
    help='Path to cache. Defaults to [data_path[1]]/cache.th7'},
    {arg='io_helper', type='table', req=true, 
    help='io_helper to helper pass the txt list'}
    )
    -- globals
    -- gm = require 'graphicsmagick'

    -- locals
    
    self:whichSet(which_set)
    self._load_size = load_size
    self._sample_size = sample_size or self._load_size
    self._verbose = verbose   
    self._classID_file = classID_file
    self._video_list = type(video_list) == 'string' and video_list
    self._data_path = type(data_path) == 'string' and {data_path} or data_path
    self.frames_per_select = frames_per_select
    self.log.info('[videoclassset] _load_size:', self._load_size)
    self.log.info('\t _sample_size;', unpack(self._sample_size))
    self.log.info('\t _verbose:', self._verbose)
    self.log.info('\t _classID_file:', self._classID_file)
    self.log.info('\t _data_list:', self._video_list)
    self.log.info('\t _data_path:', self._data_path[1])
    self.log.info('\t get sample_func:', self.sample_func)
    helper.CheckFileExist(self._video_list)
    helper.CheckFileExist(self._data_path[1])
    self._sample_func = sample_func
    -- TODO: assert input argument sample_func is valid
    self._sort_func = sort_func
    self._cache_mode = cache_mode
    self._cache_path = cache_path or paths.concat(self._data_path[1], 'cache.th7')
    self.io_helper = io_helper 
    -- indexing and caching
    assert(_.find({'writeonce','overwrite','nocache','readonly'}, cache_mode), 'invalid cache_mode :'..cache_mode)
    local cacheExists = paths.filep(self._cache_path)
    self.log.info('[VideoClassSet] cache_mode: ', cache_mode)
    if cache_mode == 'readonly' or (cache_mode == 'writeonce' and cacheExists) then
        if not cacheExists then
            error"'readonly' cache_mode requires an existing cache, none found"
        end
        self.log.info('\t cacheExists, loadIndex')
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
    for i, k in ipairs{'classList', '_classes', '_classIndices', 'classListVideo', 'classListVideoIndices', 'classListFrameIndices', 'videoList', 'videoIndices', 'videoLength', 'videoPath'} do
        index[k] = self[k]
    end
    torch.save(self._cache_path, index)
    self.log.info(string.format('\t saveIndex checking #classList %d, #_classes %d, #_classIndices %d,'..
            ' #classListVideo %d, #classListVideoIndices %d, #classListFrameIndices %d, '..
            ' #videoList %d, #videoIndices %d, #videoLength %d, #videoPath %d', 
            #index.classList, #index._classes, #index._classIndices,
            #index.classListVideo, #index.classListVideoIndices, #index.classListFrameIndices,
            #index.videoList, #index.videoIndices, #index.videoLength, #index.videoPath))
    self.log.info('\t saveIndex in ', self._cache_path, ' after build')
end

function VideoClassSet:loadIndex()
    local index = torch.load(self._cache_path)
    for k, v in pairs(index) do
        self[k] = v
    end
    self._n_sample = #self.videoPath
    self.log.info('load index from '..self._cache_path..' done. get _n_sample ', self._n_sample)

    self.log.info(string.format('\t loadIndex checking #classList %d, #_classes %d, #_classIndices %d,'..
            ' #classListVideo %d, #classListVideoIndices %d, #classListFrameIndices %d, '..
            ' #videoList %d, #videoIndices %d, #videoLength %d, #videoPath %d', 
            #self.classList, #self._classes, #self._classIndices,
            #self.classListVideo, #self.classListVideoIndices, #self.classListFrameIndices,
            #self.videoList, #self.videoIndices, #self.videoLength, #self.videoPath))
end

----
-- index_class, index of class in classes table
-- classes, table of all class name
--      key: index_class, value: class name
-- classIndices, table pf index of class
--      key: class name, value: index_class
-- classListVideo, table of (table of video for each class), 
--      key: index_class, value: table of video titles
-- videoIndices, table of video's index in the videoList
--      key: title, value: index_class
-- videoList, table for all videos of from all class
--      key: index_class, value: title of video
-- videoPath, table of all videos' paths
--      key: index_video, value: path of video, can be the '.bin' file or the video folder
-- videoLength, table of length of all videos, key: title
--      key: title, value: length of the video
-- classListVideoIndices
--      key: index_class, value: torch.Tensor/length=num_video/contain index_video
-- videoClass: torch.Tensor 
--      key: index_video, value: index_class
-----
function VideoClassSet:buildIndex()
    -- loop over each paths folder, get list of unique class names, 
    -- also store the directory paths per class
    -- {'_videoes','_videoIndices', '_videolabel', 'imagePath','imageClass','classList','classListSampleTitle'} do
    -- local classes = {} -- {'v_aas': true, 'cv_df': true, ...}
    -- local classList = {} -- {'v_applysss', 'v_running', 'v_fdf'}

    local classList = {} 
    local classes = {}
    local classListVideo = {}
    local classIndices = {}
    local videoList = {}
    local videoIndices = {}
    local videoLabel = {}
    local videoLength = {}
    local videoPath = {}


    local data_dict = self.io_helper.ReadGTText2DataDict(
        self._classID_file, self._video_list, 
        classList, classes, classListVideo, classIndices,
        videoList, videoIndices, videoLabel, videoLength, videoPath)

    self.classList = classList 
    self._classes = classes
    self._classIndices = classIndices
    self.classListVideo = classListVideo
    self.videoList = videoList
    self.videoIndices = videoIndices
    self.videoLength = videoLength
    self.videoPath = videoPath
    
    local runningIndex = 0
    self.videoClass = torch.Tensor(#self.videoList):fill(0)
    self.classListVideoIndices = {}
    for index_class = 1, #self._classes do
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
    self.log.info('building classListVideoIndices done, get size ', runningIndex)

    runningIndex = 0
    self.classListFrameIndices = {}
    self.framesClass = torch.Tensor():resize(torch.Tensor(self.videoLength):sum())
    for index_video = 1, #self.videoList do
        local num_frames = self.videoLength[index_video]
        assert(num_frames >0)
        self.classListFrameIndices[index_video] = torch.linspace(
        runningIndex+1, runningIndex+num_frames, num_frames)
        self.framesClass[{{runningIndex + 1, runningIndex + num_frames}}]:fill(self.videoClass[index_video])
        runningIndex = runningIndex + num_frames
    end

    if self._verbose then
        self.log.info("found " .. #self._classes .. " classes")
    end

    ---------------------------------------------------------------------
    -- find the image path names
    self._n_sample = #self.videoList
    self._n_video = self.videoClass:size(1)
    self._n_frame = self.framesClass:size(1)
    ---------------------------------------------------------------------
    if self._verbose then
        self.log.info('Updating classList and videoLabel appropriately')
    end
    self.log.info(string.format('[VideoClassSet] buildIndex done, get #samples %d, #video %d, #frames %d intotal', self._n_sample, self._n_video, self._n_frame))
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

function VideoClassSet:sub(batch, start, stop)
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
            local videoPath = ffi.string(torch.data(self.videoPath[idx]))
            local dst = self:getImageBuffer(i)
            -- pass video path to sample function
            dst = sampleFunc(self, dst, videoPath) 
            table.insert(inputTable, dst)
            table.insert(targetTable, self.videoClass[idx])     
            i = i + 1
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
        return batch
    end
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
        local imgpath = ffi.string(torch.data(self.videoPath[idx]))
        local dst = self:getImageBuffer(i)
        dst = sampleFunc(self, dst, imgpath)
        table.insert(inputTable, dst)
        table.insert(targetTable, self.videoClass[idx])
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
        targetTensor[i] = targetTable[i]
    end
    assert(inputTensor)
    assert(targetTensor)
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
    local video_path = paths.concat(self._data_path[1], path)
    self.log.info('[loadVideo] ', video_path)
    local concat = torch.load(video_path)
    local data = concat.data
    assert(data)
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

-- driver to call the specific sample_function, in a batch_size view
-- Sample a class uniformly, and then uniformly samples example from class.
-- This keeps the class distribution balanced.
-- sampleFunc is a function that generates one or many samples
-- from one image. e.g. sampleDefault, sampleTrain, sampleTest.
function VideoClassSet:sample(batch, nSample, sampleFunc)
    self.log.trace('VideoClassSet sampling ')
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
    self.log.trace('\t nSample =', nSample)
    local inputTable = {}
    local targetTable = {}   
    for i=1, nSample do
        -- sample class(label)
        local index_class = torch.random(1, #self._classes)
        self.log.trace('select index_class ', index_class, ' has video: ', #self.classListVideo[index_class])
        -- sample video from class
        local index_in_class = torch.random(1, #self.classListVideo[index_class])
        local index_video = self.classListVideoIndices[index_class][index_in_class]
        self.log.trace('select index_in_class ', index_in_class, ' index_video: ', index_video)
        -- local videoPath = ffi.string(torch.data(torch.CharTensor({self.videoPath[self.classListVideoIndices[index_class][index_in_class]]})))
        
        local videoPath = self.videoPath[index_video]

        self.log.trace('get videopath: ', videoPath)
        local dst = self:getImageBuffer(i)
        dst = sampleFunc(self, dst, videoPath)
        table.insert(inputTable, dst)
        table.insert(targetTable, index_class)  
    end

    local inputView = batch and batch:inputs() or dp.VideoView()
    local targetView = batch and batch:targets() or dp.ClassView()
    local inputTensor = inputView:input() or torch.FloatTensor()
    local targetTensor = targetView:input() or torch.IntTensor()

    inputTensor, targetTensor = self:tableToTensor(inputTable, targetTable, inputTensor, targetTensor)

    -- assert(inputTensor:size(2) == 3)
    assert(inputTensor:size(1) == #inputTable)
    assert(targetTensor:size(1) == #targetTable)
    self.log.trace('tableToTensor return: size ', helper.PrintSize(inputTensor))
    assert(inputView.isView)
    self.log.trace('calling dataview forward')
    inputView:forward('btchw', inputTensor)
    targetView:forward('b', targetTensor)
    targetView:setClasses(self._classes)
    assert(batch:inputs(inputView))
    assert(batch:targets(targetView))  

    collectgarbage()
    self.log.trace('[sample] done, return batch')
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
    local frames_per_select 
    if path then
        -- if given dst tensor, inder frames_per_select from the first dimension
        if torch.isTensor(dst) and dst:dim() ~= 0 then
            frames_per_select = dst:size(1)
        else 
            frames_per_select = self.frames_per_select
        end
        assert(frames_per_select, 'frames_per_select must be set!')
    end
    if not path then
        path = dst
        dst = torch.FloatTensor()
    end
    if not dst then
        dst = torch.FloatTensor()
    end
    local out = self:loadVideo(path)
    -- print(out)
    -- out = out:view(out:size(1), unpack(self._load_size))
    dst:resize(frames_per_select, unpack(self._sample_size)):fill(0)
    local num_frames = #out
    local index_start = torch.random(1, num_frames)
    local i = 1
    for index = index_start, math.min(index_start + frames_per_select - 1, num_frames) do
        dst[i]:copy(out[index]:view(unpack(self._load_size)))
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
    self:sampleVolume(dst, path)
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
        require 'dprnn'
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
    local inputPointer = tonumber(ffi.cast('intptr_t', torch.pointer(input:storage())))
    local targetPointer = tonumber(ffi.cast('intptr_t', torch.pointer(target:storage())))
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
end
