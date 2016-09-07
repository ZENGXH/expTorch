local UCF101, parent = torch.class("dp.UCF101", "dp.DataSource")
UCF101._name = 'UCF101'
UCF101._image_axes = 'bchw'
UCF101._classes = torch.range(1,1000):totable()

function UCF101:__init(config)
    config.name = config.name or 'ds-ucf101'
    parent.__init(self, config)
    config = config or {}
    assert(torch.type(config) == 'table' and not config[1], 
    "Constructor requires key-value arguments")
    if config.input_preprocess or config.output_preprocess then
        error("UCF101 doesnt support Preprocesses. "..
        "Use Sampler ppf arg instead (for online preprocessing)")
    end
    local load_all, input_preprocess, target_preprocess
    local args = {} 
    dp.helper.unpack_config(args, {config},
        'UCF101',
        'ucf101 action regonition dataset',
        {arg='classID_file', type='string', req=true, 
        help='dataList contain all the filename, length, label'},
        {arg='data_test_list', type='string', req=true, help='list for all video'},
        {arg='data_train_list', type='string', req=true, help='list for all video'},
        {arg='shuffle_train_data', type='boolean', req=true, help='shuffle_train_data if true'},
        {arg='shuffle_test_data', type='boolean', req=true, help='shuffle_train_data if true'},
        {arg='data_root', type='string', default=dp.DATA_DIR,
        help='root of datadir'},
        {arg='data_folder', type='string', req=true, help='folder name of the datas'},
        {arg='load_size', type='table', default={1, 1024, 1, 1},
        help='an approximate size to load the feature from bin file, '..
        'including batchSize=1. Defaults to 1 x 1024 x 1 x 1.'},
        {arg='sample_size', type='table', default={1024, 1, 1},
        help='a consistent size for cropped patches from loaded images'..
        'without batchSize, without num_frames Defaults to .'},
        {arg='input_type', type='string', help='flow or rgb'},
        {arg='frames_per_select_train', type='number', req=true},
        {arg='frames_per_select_test', type='number'},
        {arg='verbose', type='boolean', default=true, help='Verbose mode during initialization'},
        {arg='load_all', type='boolean', 
        help='Load all datasets : train and valid', default=true},
        {arg='is_test_only', type='boolean', default=false,
        help='do not run train if true'},
        {arg='cache_mode', type='string', default='writeonce',
        help='writeonce : read from cache if exists, else write to cache. '..
        'overwrite : write to cache, regardless if exists. '..
        'nocache : dont read or write from cache. '..
        'readonly : only read from cache, fail otherwise.'}
        )
     
    local load_all = args.load_all
    local input_type = args.input_type
    local data_folder = args.data_folder
    local data_root = args.data_root
    self._classID_file = args.classID_file 
    self._data_test_list = args.data_test_list
    self._data_train_list = args.data_train_list
    self._load_size = args.load_size or {1, 1024, 1, 1}
    self._sample_size = args.sample_size or {1024, 1, 1} -- do not consider frames_per_select_train in sample_size
    self._verbose = args.verbose
    self.frames_per_select_train = args.frames_per_select_train
    self.frames_per_select_test = args.frames_per_select_test or args.frames_per_select_train -- if not set, same as train
    self._cache_mode = args.cache_mode
 
    self._image_size = self._sample_size
    self.log:info('frames_per_select_train: ', args.frames_per_select_train, ' frames_per_select_test ', self.frames_per_select_test)
    self.log.info('ucf101 load_size: ', unpack(self._load_size), ' sample_size ', unpack(self._sample_size), 'image_size', unpack(self._image_size))
    self._feature_size = self._sample_size[1]*self._sample_size[2]*self._sample_size[3]
    
    self._train_path=paths.concat(data_root, 'ucf101_'..input_type, 'ucf101_'..input_type..'_train', data_folder)
    self._test_path=paths.concat(data_root, 'ucf101_'..input_type, 'ucf101_'..input_type..'_test', data_folder)
    self._meta_path=paths.concat(data_root, 'ucf101_'..input_type, 'metadata', data_folder)
    self._io_helper=loadfile(paths.concat(dp.DPRNN_DIR, 'utils', 'ucf101_helper.lua'))()

    self.is_test_only = args.is_test_only
    if load_all then
        self.log.trace('[ds] UCF101 load all:Train, Valid and Meta')
        if not self.is_test_only then
            self:loadTrain()
        end
        self:loadTest()
        -- self:loadValid()
        -- self:loadMeta()
    else
        self.log.info('not load all, call loadTrain | loadValid | youself')
    end
    if args.shuffle_train_data then
        self:ShuffleAttributeSet('train')
    end
    if args.shuffle_test_data then
        self:ShuffleAttributeSet('test')
    end
    self.log.info('[UCF101 DataSource] init done')
end

function UCF101:loadTrain()
    assert(self._data_train_list)
    self.log.trace('loadTrain', self.frames_per_select_train)
    local dataset = dp.VideoClassSet{
        name='ucf-trainSet',
        classID_file=self._classID_file,
        data_list=self._data_train_list,
        data_path=self._train_path, 
        load_size=self._load_size,
        which_set='train', 
        sample_size=self._sample_size,
        verbose=self._verbose, 
        frames_per_select=self.frames_per_select_train,
        sample_func='sampleTrain',
        sort_func=function(x,y)
            return tonumber(x:match('[0-9]+')) < tonumber(y:match('[0-9]+'))
        end, 
        cache_mode=self._cache_mode,
        io_helper=self._io_helper
    }
    self:trainSet(dataset)
    self.log.info('[ds] UCF101 loadTrain done')
    return dataset
end
------------------------------------------------------------------
-- VideoClassSet support shuffle
------------------------------------------------------------------
function UCF101:ShuffleAttributeSet(attribute)
    if not self:GetAttributeSet(attribute) then
        self.log.fatal('empty dataset ', attribute)
        return 
    else
       self:GetAttributeSet(attribute):Shuffle()
   end
end
function UCF101:loadValid()
    self.log.trace('loadvalid')
    assert(self.frames_per_select_train)
    local dataset = dp.VideoClassSet{
        name='ucf-validSet',
        classID_file=self._classID_file,
        data_list=self._data_test_list,
        data_path=self._valid_path, 
        load_size=self._load_size,
        which_set='valid', 
        sample_size=self._sample_size,
        verbose=self._verbose, 
        frames_per_select=self.frames_per_select_train,
        sample_func='sampleTest',
        sort_func=function(x,y)
            return tonumber(x:match('[0-9]+')) < tonumber(y:match('[0-9]+'))
        end, 
        cache_mode=self._cache_mode,
        io_helper=self._io_helper
    }
    self.log.info('UCF101 loadValid done')
    self:validSet(dataset)
    return dataset
end

function UCF101:loadTest()
    self.log.trace('loadTest')
    local dataset = dp.VideoClassSet{
        name='ucf-testSet',
        classID_file=self._classID_file,
        data_list=self._data_test_list,
        data_path=self._test_path, 
        load_size=self._load_size,
        which_set='test', 
        sample_size=self._sample_size,
        verbose=self._verbose, 
        frames_per_select=self.frames_per_select_test,
        sample_func='sampleTest',
        sort_func=function(x,y)
            return tonumber(x:match('[0-9]+')) < tonumber(y:match('[0-9]+'))
        end, 
        cache_mode=self._cache_mode,
        io_helper=self._io_helper
    }
    self.log.info('UCF101 loadValid done')
    self:testSet(dataset)
    return dataset
end


function UCF101:loadMeta()
    local classInfoPath = paths.concat(self._meta_path, 'classInfo.th7')
    if paths.filep(classInfoPath) then
        self.classInfo = torch.load(classInfoPath)
    else
        if self._verbose then
            print("UCF101: skipping "..classInfoPath)
            print("To avoid this message use harmonizeimagenet.lua "..
            "script and pass correct meta_path")
        end
    end
end

-- Returns normalize preprocessing function (PPF)


function UCF101:multithread(nThread, batch_size)
    self.log.info('[DataSource UCF101] calling multithread')
    if self._train_set then
        self._train_set:multithread(nThread, batch_size)
    end
    if self._valid_set then
        self._valid_set:multithread(nThread, batch_size)
    end
    if self._test_set then
        self._test_set:multithread(nThread, batch_size)
    end
    self.log.info('[DataSource UCF101] multithread set as ', nThread)
end
