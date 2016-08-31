------------------------------------------------------------------------
--[[ videoclassset ]]--
-- a dataset for video  in a flat folder structure :
-- [data_path]/[vdieo]/[framesname].jpeg  (folder-name is video-name)
-- optimized for extremely large datasets (9000 videos+).
-- video on disk can have different length.
-- video can be reach by reading the whole video ot frames bin
-- currently not modify for frames-level-raw-video-frames input
------------------------------------------------------------------------

local VideoClassSet, parent = torch.class("dp.VideoClassSet", "dp.ImageClassSet")
-- batch, timeStep, channels, height, width
VideoClassSet._input_shape = 'btchw' 
VideoClassSet._output_shape = 'b'
VideoClassSet.isVieoClassSet = true
VideoClassSet.isImageClassSet = true
function VideoClassSet:__init(config)

    assert(type(config) == 'table', "Constructor requires key-value arguments")
    local args = {} 
    dp.helper.unpack_config(args,
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

    {arg='frames_per_select', type='number', req=true,
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
    {arg='io_helper', type='table', req=true, default=dofile(paths.concat(dp.DPRNN_DIR, 'utils', 'ucf101_helper.lua')), 
    help='io_helper to helper pass the txt list'}
    )
    -- globals
    -- gm = require 'graphicsmagick'

    -- locals
    
    self:whichSet(args.which_set)
    self._load_size = args.load_size
    self._sample_size = args.sample_size or self._load_size
    self._verbose = args.verbose   
    self._classID_file = args.classID_file
    self._data_list = type(args.data_list) == 'string' and args.data_list
    self._data_path = type(args.data_path) == 'string' and {args.data_path} or args.data_path
    self.frames_per_select = args.frames_per_select
    dp.helper.CheckFileExist(self._data_list)
    dp.helper.CheckFileExist(self._data_path[1])
    self._sample_func = args.sample_func
    -- TODO: assert input argument sample_func is valid
    self._sort_func = args.sort_func
    local cache_mode = args.cache_mode
    self._cache_mode = args.cache_mode
    self._cache_path = args.cache_path or paths.concat(self._data_path[1], 'cache.th7')
    self.io_helper = args.io_helper 
    self.JoinTable = nn.JoinTable(1)
    assert(self.io_helper)
    -- indexing and caching
    assert(_.find({'writeonce','overwrite','nocache','readonly'}, args.cache_mode), 'invalid cache_mode :'..args.cache_mode)

    parent.__init(self, config)
    self.log.info('[videoclassset] _load_size:', self._load_size,
    '\t frames_per_select', self.frames_per_select,
    '\t _sample_size;', unpack(self._sample_size),
    '\t _verbose:', self._verbose,
    '\t _classID_file:', self._classID_file,
    '\t _data_list:', self._data_list,
    '\t _data_path:', self._data_path[1],
    '\t get sample_func:', self.sample_func)
 
    -- self.log.info('__init VideoClassSet', unpack(config))
    -- build index in parent
    --[[
    local cacheExists = paths.filep(self._cache_path)
    if cache_mode == 'readonly' or (args.cache_mode == 'writeonce' and args.cacheExists) then
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
    self._config = args.config 
    ]]--
    self._class_set = 'VideoClassSet' 
    assert(self._input_shape == VideoClassSet._input_shape)
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
        self._classID_file, self._data_list, 
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
   self.log.tracefrom('request batch with size ', batch_size, ' sample_size(input) ', unpack(self._sample_size))
    return dp.Batch{
        which_set=self._which_set,
        inputs=dp.VideoView(self._input_shape, 
            torch.FloatTensor(batch_size, self.frames_per_select, unpack(self._sample_size))),
        targets=dp.ClassView(self._output_shape, 
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
    if (not batch or batch == nil) or (not stop or stop == nil) then
        if batch or batch ~= nil then -- first arg exist, third not exist
            stop = start
            start = batch
        end
        self.log.trace('building batch with size ', self:nSample())

        -- inie a batch nil
        batch = dp.Batch{
            which_set=self:whichSet(), 
            epoch_size=self:nSample()
        }
   end

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
    inputView:forward(self._input_shape, inputTensor)
    targetView:forward(self._output_shape, targetTensor)
    targetView:setClasses(self._classes)
    batch:inputs(inputView)
    batch:targets(targetView)
    return batch
end

-- converts a table of samples (and corresponding labels) to tensors
-- different with ImageClassSet, cause multi-framesPerDraw only corresponding to 1 output
function VideoClassSet:tableToTensor(inputTable, targetTable, inputTensor, targetTensor)
    -- inputTensor possible shape: (t, c, h, w) or (t, c) or (1, c, h, w) or (1, c) or (c) or (c, h, w)
    inputTensor = inputTensor or torch.FloatTensor()
    targetTensor = targetTensor or torch.IntTensor()
    local n = #targetTable -- batchSize

    inputTensor:resize(n, self.frames_per_select, unpack(self._sample_size)):copy(self.JoinTable:forward(inputTable))
    targetTensor:resize(n)
    --[[
    for i = 1, #inputTable do
        inputTable[i] = inputTable[i]:view(1, self.frames_per_select, unpack(self._sample_size))
        targetTensor[i] = targetTable[i]
    end
    inputTensor:copy(nn.JoinTable(1):forward(inputTable))
    ]]--
     
    for i = 1, n do
        targetTensor[i] = targetTable[i]
    end
    
    assert(inputTensor)
    assert(targetTensor)
    return inputTensor, targetTensor
end

function VideoClassSet:loadImage(path)
    return self:loadVideo(path)
end

-- return a table contain all frames of the video 
function VideoClassSet:loadVideo(path)
    local video_path = paths.concat(self._data_path[1], path)
    self.log.trace('[loadVideo] ', video_path)
    local concat = torch.load(video_path)
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
        dst = sampleFunc(self, dst, videoPath) -- enlarge size from 4d to 5d
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
    self.log.trace('tableToTensor return: size ', dp.helper.PrintSize(inputTensor))
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
    local copy_end = math.min(index_start + frames_per_select - 1, num_frames)
    local copy_frames = copy_end - index_start + 1
    local narrowTable = nn.NarrowTable(index_start, copy_end-index_start+1) -- offset, length
    local output_select = narrowTable:forward(out)

    -- if out[1]:dim() == 1 then -- so sad we can not use Join
    for i = 1, #output_select do
        output_select[i] = output_select[i]:view(1, unpack(self._sample_size)) -- all from 1024 -> 1, 1024, 1, 1
    end
    dst:narrow(1, 1, copy_frames):copy(self.JoinTable:forward(output_select))

    return dst:view(1, frames_per_select, unpack(self._sample_size)) -- 5D
end

-- function to load the image, jitter it appropriately (random crops etc.)
function VideoClassSet:sampleTrain(dst, path)
    self:sampleVolume(dst, path)
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

function VideoClassSet:sampleAsyncPut(batch, nSample, sampleFunc, callback)
   self.log.info('[sampleAsyncPut] with view in ', self._input_shape, ' for nSample ', nSample)
   self._iter_mode = self._iter_mode or 'sample'
   if (self._iter_mode ~= 'sample') then
      error'can only use one Sampler per async ImageClassSet (for now)'
   end  
   
   if not batch or batch == nil then
      self.log.trace('batch is nil size of buffer_batches: ', self._buffer_batches:length() )
      batch = (not self._buffer_batches:empty()) and self._buffer_batches:get() or self:batch(nSample)
      self.log.trace('batch is now not nil')
   else
       self.log.trace('batch is nil')
   end

   local input = batch:inputs():input()
   local target = batch:targets():input()
   print(input:size(), input:dim())
   assert(input:dim() == 5, 'get input dim: i')
   assert(target)
   
   local p = torch.pointer(input:storage()) 
   -- transfer the storage pointer over to a thread
   local targetPointer = tonumber(ffi.cast('intptr_t', 
        torch.pointer(target:storage())))
 
   self.log.trace('get target pointer')
   local inputPointer = tonumber(ffi.cast('intptr_t', 
        torch.pointer(input:storage())))
   self.log.trace('get input pointer')
   input:cdata().storage = nil
   target:cdata().storage = nil
   
   self._send_batches:put(batch)
    
   self.log.trace('put batch')
   assert(self._threads:acceptsjob())
   self.log.trace('start add job')
   self._threads:addjob(
      -- the job callback (runs in data-worker thread)
      function()
         -- set the transfered storage
         
         print('setStorage')
         torch.setFloatStorage(input, inputPointer)
         torch.setIntStorage(target, targetPointer)
         local view =  'btchw'
         
         tbatch:inputs():forward(view, input)
         tbatch:targets():forward('b', target)

         print('forward')
         
         dataset:sample(tbatch, nSample, sampleFunc)
         assert(tbatch:inputs():input()) 
         assert(tbatch:targets():input()) 
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
         batch:inputs():forward('btchw', input)
         batch:targets():forward('b', target)
         callback(batch)
         batch:targets():setClasses(self._classes)
         -- self.log.trace('putting to _recv_batches: ', #self._recv_batches)
         self._recv_batches:put(batch)
         
      end
   )
end
