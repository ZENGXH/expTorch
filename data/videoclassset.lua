------------------------------------------------------------------------
--[[ videoclassset ]]--
-- a dataset for video  in a flat folder structure :
-- [data_path]/[vdieo]/[framesname].jpeg  (folder-name is video-name)
-- optimized for extremely large datasets (9000 videos+).
-- video on disk can have different length.
-- video can be reach by reading the whole video ot frames bin
-- currently not modify for frames-level-raw-video-frames input
--
-- data is filled when: sampleTrain or sampleTest is called with path
------------------------------------------------------------------------

local VisualDataSet, parent = torch.class("dp.VideoClassSet", "dp.VisualDataSet")

-- batch, timeStep, channels, height, width
VisualDataSet._input_shape = 'btchw' 
VisualDataSet._output_shape = 'b'
VisualDataSet.isVieoClassSet = true
function VisualDataSet:__init(config)
    parent.__init(self, config)
    assert(type(config) == 'table', "Constructor requires key-value arguments")
    local args = {} 
    dp.helper.unpack_config(args,
    {config},
    'VisualDataSet', 
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
    -- used for VisualDataSet:tableToTensor(...)
    self._input_view_type='dp.VideoView'
    self._target_view_type='dp.ClassView'
    self._input_shape_set = {self.frames_per_select, unpack(self.sample_size)}
    self._target_shape_set = {batch_size}
    self._target_view_tensor = 'IntTensor'
    self._input_view_tensor = 'FloatTensor'

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
 
    -- self.log.info('__init VisualDataSet', unpack(config))
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
    self._class_set = 'VisualDataSet' 
    assert(self._input_shape == VisualDataSet._input_shape)
end

function VisualDataSet:saveIndex()
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

function VisualDataSet:loadIndex()
    local index = torch.load(self._cache_path)
    for k, v in pairs(index) do
        self[k] = v
    end
    self._n_sample = #self.videoPath
    self.log.info('load index from '..self._cache_path..
        ' done. get _n_sample ', self._n_sample)
    self.log.info(string.format('\t loadIndex checking #classList %d, '..
        '#_classes %d, #_classIndices %d,'..
        ' #classListVideo %d, #classListVideoIndices %d, '..
        '#classListFrameIndices %d, '..
        ' #videoList %d, #videoIndices %d, #videoLength %d, #videoPath %d', 
         #self.classList, #self._classes, #self._classIndices,
         #self.classListVideo, #self.classListVideoIndices, 
         #self.classListFrameIndices, #self.videoList, 
         #self.videoIndices, #self.videoLength, #self.videoPath))
end


----------------------------------------------------------------------
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
--------------------------------------------------------------------

function VisualDataSet:buildIndex()
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
    self.log.info(string.format('[VisualDataSet] buildIndex done, get #samples %d, #video %d, #frames %d intotal', self._n_sample, self._n_video, self._n_frame))
end

function VisualDataSet:FillSubViewWithIndex()
    error('not support')
end

function 
function VisualDataSet:sub(batch, start, stop)
    error('not support')
end

-------------------------------------------------------------------
-- FillBatch.. is responsible to fill a batch, it need to loop by the
-- batch_size, dicide the index/start-end point of each video
function VisualDataSet:FillBatchWithSub(batch, start, stop)
    error('not spport')
end

function VisualDataSet:index(batch, indices)
    error('not support')
end

-------------------------------
-- <override> do the combining of tensor in to a batch in the subclass since
-- then the output and input shape may be different
--
-- converts a table of samples (and corresponding labels) to tensors
-- different with ImageClassSet, cause multi-framesPerDraw only 
-- corresponding to 1 output
-- 
-- @param inputTable
-- @orarm outputTable
-- @oaram inputTensor
-- @oaram outputTensor
--
-- @return inputTensor
-- @return outputTensor
----------------------------------
function VisualDataSet:tableToTensor(inputTable, targetTable, inputTensor, targetTensor)
    -- inputTensor possible shape: (t, c, h, w) or (t, c) or (1, c, h, w) or (1, c) or (c) or (c, h, w)
    -- inputTensor = torch.FloatTensor()
    -- targetTensor = targetTensor or torch.IntTensor()
    local n = #targetTable -- batchSize
    inputTensor:copy(self.JoinTable:forward(inputTable))
    assert(targetTensor:size(1) == n and targetTensor:dim() == 1)
    
    for i = 1, n do
        targetTensor[i] = targetTable[i]
    end
    
    assert(inputTensor)
    assert(targetTensor)
    assert(inputTensor:size(1) == #inputTable)
    assert(targetTensor:size(1) == #targetTable)
    self.log.trace('tableToTensor return: size ', 
        dp.helper.PrintSize(inputTensor))

    return inputTensor, targetTensor
end


-------------------------------------------------------------
--[[ acture sample implement ]]--
-- this function is working in the worker
-- get input Tensor from tbatch input and target view, fill the data load from
-- disk, then put the tensor back to tbatch to the worker
-- @param batch, which is filled 
-- @param start, stop: the start video_index and end video_index
-- @param callback
--
-- return batch which contain the sample data
-------------------------------------------------------------
function VisualDataSet:FillBatchOrderSample(tbatch, start, stop)
    assert(tbatch:IsFilled())
    local inputTable = {}
    local targetTable = {}       
    inputTable, targetTable = self:_GetOrderSample(start, stop, inputTable, targetTable)
    return self:_FillBatch(tbatch, inputTable, targetTable)
end

function VisualDataSet:FillBatchRandomSample(tbatch, nSample)
    assert(tbatch:IsFilled())
    local inputTable = {}
    local targetTable = {}       
    inputTable, targetTable = self:_GetRandomSample(nSample, inputTable, targetTable)
    return self:_FillBatch(tbatch, inputTable, targetTable)
end

-------------- [private method] ----------------------------------
-- copy the selected data in the inputTable into tBatch tensor
------------------------------------------------------------------
function VisualDataSet:_FillBatch(tbatch, inputTable, targetTable)
    assert(batch:IsFilled())
    local inputTensor = tbatch:GetView('input'):GetInputTensor()
    local targetTensor = tbatch:GetView('target'):GetInputTensor()
    inputTensor, targetTensor = self:tableToTensor(inputTable, 
        targetTable, inputTensor, targetTensor)
    -- assert(inputTensor:size(2) == 3)
    -- batch:SetView('input', dp.VideoView
    self.log.trace('calling dataview forward')
    tbatch:GetView('input'):forwardPut(self._input_shape, inputTensor)
    tbatch:GetView('target'):forwardPut(self._output_shape, targetTensor)
    tbatch:GetView('target'):setClasses(self._classes)
    collectgarbage()
    self.log.trace('[sample] done, return batch')
    return tbatch
end

function VisualDataSet:_GetRandomSample(nSample, inputTable, targetTable)
  for i = 1, nSample do
        -- sample class(label)
        local index_class = torch.random(1, #self._classes)
        self.log.trace('select index_class ', index_class, 
            ' has video: ', #self.classListVideo[index_class])
        -- sample video from class
        local index_in_class_ran = torch.random(1, 
            #self.classListVideo[index_class])
        local index_video = self.classListVideoIndices[
            index_class][index_in_class_ran]
        self.log.trace('select index_in_class_ran ', index_in_class_ran, 
            ' index_video: ', index_video)
        
        local videoPath = self.videoPath[index_video]
        self.log.trace('get videopath: ', videoPath)

        local dst = self:getImageBuffer(i)
        dst = sampleFunc(self, dst, videoPath) 
        -- enlarge size from 4d to 5d
        table.insert(inputTable, dst)
        table.insert(targetTable, index_class)  
    end
    return inputTable, targetTable
end

function VisualDataSet:_GetOrderSample(start, stop, inputTable, targetTable)
  for i = 1, start - stop + 1 do
        -- sample class(label)
        local index_class = i - 1 + start
        self.log.trace('select index_class ', index_class, 
            ' has video: ', #self.classListVideo[index_class])
        -- sample video from class
        local index_in_class_ran = torch.random(1, 
            #self.classListVideo[index_class])
        local index_video = self.classListVideoIndices[
            index_class][index_in_class_ran]
        self.log.trace('select index_in_class_ran ', index_in_class_ran, 
            ' index_video: ', index_video)
        
        local videoPath = self.videoPath[index_video]
        self.log.trace('get videopath: ', videoPath)

        local dst = self:getImageBuffer(i)
        dst = sampleFunc(self, dst, videoPath) 
        -- enlarge size from 4d to 5d
        table.insert(inputTable, dst)
        table.insert(targetTable, index_class)  
    end
    return inputTable, targetTable
end


-- driver to call the specific sample_function, in a batch_size view
-- Sample a class uniformly, and then uniformly samples example from class.
-- This keeps the class distribution balanced.
-- sampleFunc is a function that generates one or many samples
-- from one image. e.g. sampleDefault, sampleTrain, sampleTest.

-- @batch: dp.Batch, must be create outside
-- @nSample: number of sample, equal to batch_size normally
-- @sample_func: string which load data function is gone be used
-- #TODO, make sample_func a func, allow define outsize?
-- @param sample_func: string, can be 
--  'LoadDataDefaultFunc' for whole video,
--  'LoadData5DRandomFunc' which self.frames_per_select, 
--  'LoadDataValidFunc' | 'LoadDataTestFunc' | 'LoadDataTrainFunc'
-- return filled batch
--------------------------------------------------------------

-- function to load the image, jitter it appropriately (random crops etc.)

function VisualDataSet:PrepareBatchBufferIfEmpty(threads, batch_size, )
    if self._buffer_batches:empty() then
        for i = 1, threads do
            local batch = self:CreateBatchWithSize(batch_size)
            self:InitBatchWithSize
            self._buffer_batches:put(batch)
        end
    end
end

------------------------------------------------------
-- load the data from path into tensor
-- ByDefault load the whole videoPath
--
-- @param dst, torch.Tensor()
-- @param videoPath, dataPath of the data
-- 
-- @return data torch.Tensor
-----------------------------------------------------
function VisualDataSet:loadImage(path)
    return self:loadVideo(path)
end

-- return a table contain all frames of the video 
function VisualDataSet:loadVideo(path)
    local video_path = paths.concat(self._data_path[1], path)
    self.log.trace('[loadVideo] ', video_path)
    local concat = torch.load(video_path)
    local data = concat.data
    return data
end

function VisualDataSet:LoadDataDefaultFunc(videoPath) i
    -- enlarge size from 4d to 5d
    -- #TODO: checking?
    return input = self:loadVideo(path)
    -- return input
end


-----------------------------------------------------------------------
-- Given video Data table, narrow it by start and end, return as a 5D Tensor
-- @param out: table, {Tensor in 1D: 1024..}
-- @param start: int
-- @param copy_end: int, copy_end - start + 1 may < self.frames_per_select
-- @param dst: output Tensor
--
-- @return dst 
------------------------------------------------------------------------
function VisualDataSet:_NarrowOutputTableToDst(out, start, copy_end, dst)
    dst = dst:resize(self.frames_per_select, unpack(self._sample_size)):fill(0)
    local narrowTable = nn.NarrowTable(index_start, 
        copy_end - index_start + 1) -- offset, length
    local output_select = narrowTable:forward(out)
    local copy_frames = copy_end - index_start + 1
    for i = 1, #output_select do
        -- add first dim 1, prepare for join to be a volume
        output_select[i] = output_select[i]:view(1, 
            unpack(self._sample_size)) -- all from 1024 -> 1, 1024, 1, 1
    end
    dst:narrow(1, 1, copy_frames):copy(
        self.JoinTable:forward(output_select))
    -- add first dim 1, prepare for join to be a batch
    return dst:view(1, self.frames_per_select, unpack(self._sample_size)) -- 5D
end

------------------------------------------------------------------------
-- return torch.Tensor in 5D(1, self.frames_per_select,
--  unpack(self_sample_size)
-- which is sub select from the original Video
-- 
-- @param path, string
-- @param start: Int
-- @param stop: int
-- 
-- @return dst in 5D view: torch.Tensor
------------------------------------------------------------------------
function VisualDataSet:LoadData5DFunc(path, start, stop, dst)
    assert(start > 0 and stop >= start)
    assert(start - stop + 1 <= self.frames_per_select)
    local out = self:LoadDataDefaultFunc(path)
    local narrowTable = nn.NarrowTable(start, start + self.frames_per_select - 1)
    local output_select = narrowTable:forward(out)

    local out = self:LoadDataDefaultFunc(path)
    self:_NarrowOutputTableToDst(out, start, out, dst)
    return dst
end

------------------------------------------------------------------------
-- return torch.Tensor in 5D(1, self.frames_per_select,
--  unpack(self_sample_size)
-- which is randomly select from the original Video
-- @param path, string
-- @tensor should be inited outsize the function 
-- @return dst in 5D view: torch.Tensor
-----------------------------------------------------------------------
function VisualDataSet:LoadData5DRandomFunc(path, dst)
    assert(self.frames_per_select, 'frames_per_select must be set!')
    local out = self:LoadDataDefaultFunc(path)
    local num_frames = #out
    local index_start_ran = torch.random(1, num_frames)
    local i = 1
    local copy_end = math.min(index_start_ran + self.frames_per_select - 1, 
        num_frames)
    dst = self:NarrowOutputTableToDst(out, index_start_ran, copy_end, dst)
    return dst
end

function VisualDataSet:sampleTest(dst, path)
    -- TODO: do 25 sampling?
    self:sampleVolume(dst, path)
    return dst
end

function VisualDataSet:sampleValid(dst, path)
    -- TODO: do 1 sampling
    return self:LoadDataValidFunc(dst, path)
end
function VisualDataSet:LoadDataValidFunc(dst, path)
    error('not implement')
end
function VisualDataSet:sampleTest(dst, path)
    -- TODO: do 1 sampling
    return self:LoadDataTestFunc(dst, path)
end
function VisualDataSet:LoadDataTestFunc(dst, path)
    error('not implement')
end
function VisualDataSet:sampleTrain(dst, path)
    -- TODO: do 1 sampling
    return self:LoadDataTrainFunc(dst, path)
end
function VisualDataSet:LoadDataTrainFunc(dst, path)
    error('not implement')
end
