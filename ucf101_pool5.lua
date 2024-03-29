
local UCF101_pool5.lua, DataSource = torch.class("dp.UCF101_pool5.lua", "dp.DataSource")

UCF101_pool5.lua._name = 'UCF101_pool5.lua'
UCF101_pool5.lua._image_axes = 'bc'
UCF101_pool5.lua._classes = torch.range(1, 101):totable()

function UCF101_pool5.lua:__init(config)
   config = config or {}
   assert(torch.type(config) == 'table' and not config[1], 
      "Constructor requires key-value arguments")
   if config.input_preprocess or config.output_preprocess then
      error("UCF101_pool5.lua doesnt support Preprocesses. "..
            "Use Sampler ppf arg instead (for online preprocessing)")
   end
   local load_all, input_preprocess, target_preprocess
        self._args, self._load_size, self._sample_size, 
      self._train_path, self._valid_path, self._meta_path, 
      self._verbose, load_all, self._cache_mode
      = xlua.unpack(
      {config},
      'UCF101_pool5.lua',
      'ILSVRC2012-14 image classification dataset',
      {arg='load_size', type='table', 
       help='an approximate size to load the features.'
       ..' Defaults to 1024.'},

      {arg='sample_size', type='table',
       help='a consistent size for cropped patches from loaded images.'
       ..' Defaults to 3x224x244.'},

      {arg='train_path', type='string', help='path to training images',
       default=paths.concat(dp.DATA_DIR, 'UCF101_pool5.lua', 'ILSVRC2012_img_train')},

      {arg='valid_path', type='string', help='path to validation images',
       default=paths.concat(dp.DATA_DIR, 'UCF101_pool5.lua', 'ILSVRC2012_img_val')},

      {arg='meta_path', type='string', help='path to meta data',
       default=paths.concat(dp.DATA_DIR, 'UCF101_pool5.lua', 'metadata')},

      {arg='verbose', type='boolean', default=true,
       help='Verbose mode during initialization'},

      {arg='load_all', type='boolean', 
       help='Load all datasets : train and valid', default=true},

      {arg='cache_mode', type='string', default='writeonce',
       help='writeonce : read from cache if exists, else write to cache. '..
       'overwrite : write to cache, regardless if exists. '..
       'nocache : dont read or write from cache. '..
       'readonly : only read from cache, fail otherwise.'}
   )
   self._load_size = self._load_size or {1024}
   self._sample_size = self._sample_size or {3, 224, 224}
   
   self._image_size = self._sample_size
   self._feature_size = self._sample_size[1]*self._sample_size[2]*self._sample_size[3]
   
   if load_all then
      self:loadTrain()
      self:loadValid()
      self:loadMeta()
   end
end

function UCF101_pool5.lua:loadTrain()
   local dataset = dp.ImageClassSet
   {
      data_path=self._train_path, 
      load_size=self._load_size,
      which_set='train', 
      sample_size=self._sample_size,
      verbose=self._verbose, 
      sample_func='sampleTrain',
      sort_func=function(x,y)
         return tonumber(x:match('[0-9]+')) < tonumber(y:match('[0-9]+'))
      end, 
      cache_mode=self._cache_mode
   }
   self:trainSet(dataset)
   return dataset
end

function UCF101_pool5.lua:loadValid()
   local dataset = dp.ImageClassSet
   {
      data_path=self._valid_path, 
      load_size=self._load_size,
      which_set='valid', 
      sample_size=self._sample_size,
      verbose=self._verbose, 
      sample_func='sampleTest',
      sort_func=function(x,y)
         return tonumber(x:match('[0-9]+')) < tonumber(y:match('[0-9]+'))
      end, 
      cache_mode=self._cache_mode
   }
   self:validSet(dataset)
   return dataset
end

function UCF101_pool5.lua:loadMeta()
   local classInfoPath = paths.concat(self._meta_path, 'classInfo.th7')
   if paths.filep(classInfoPath) then
      self.classInfo = torch.load(classInfoPath)
   else
      if self._verbose then
         print("UCF101_pool5.lua: skipping "..classInfoPath)
         print("To avoid this message use harmonizeimagenet.lua "..
               "script and pass correct meta_path")
      end
   end
end

-- Returns normalize preprocessing function (PPF)
-- Estimate the per-channel mean/std on training set and caches results
function UCF101_pool5.lua:normalizePPF()
   local meanstdCache = paths.concat(self._meta_path, 'meanstd.th7')
   local mean, std
   if paths.filep(meanstdCache) then
      local meanstd = torch.load(meanstdCache)
      mean = meanstd.mean
      std = meanstd.std
      if self._verbose then
         print('Loaded mean and std from cache.')
      end
   else
      local tm = torch.Timer()
      local trainSet = self:trainSet() or self:loadTrain()
      local nSamples = 10000
      if self._verbose then
         print('Estimating the mean,std (per-channel, shared for all pixels) over ' 
               .. nSamples .. ' randomly sampled training images')
      end
      
      mean = {0,0,0}
      std = {0,0,0}
      local batch
      for i=1,nSamples,100 do
         batch = trainSet:sample(batch, 100)
         local input = batch:inputs():forward('bchw')
         for j=1,3 do
            mean[j] = mean[j] + input:select(2,j):mean()
            std[j] = std[j] + input:select(2,j):std()
         end
      end
      for j=1,3 do
         mean[j] = mean[j]*100 / nSamples
         std[j] = std[j]*100 / nSamples
      end
      local cache = {mean=mean,std=std}
      torch.save(meanstdCache, cache)
      
      if self._verbose then
         print('Time to estimate:', tm:time().real)
      end
   end
   
   if self._verbose then
      print('Mean: ', mean[1], mean[2], mean[3], 'Std:', std[1], std[2], std[3])
   end
   
   local function ppf(batch)
      local inputView = batch:inputs()
      assert(inputView:view() == 'bchw', 'UCF101_pool5.lua ppf only works with bchw')
      local input = inputView:input()
      for i=1,3 do -- channels
         input:select(2,i):add(-mean[i]):div(std[i]) 
      end
      return batch
   end

   if self._verbose then
      -- just check if mean/std look good now
      local trainSet = self:trainSet() or self:loadTrain()
      local batch = trainSet:sample(100)
      ppf(batch)
      local input = batch:inputs():input()
      print('Stats of 100 randomly sampled images after normalizing. '..
            'Mean: ' .. input:mean().. ' Std: ' .. input:std())
   end
   return ppf
end

function UCF101_pool5.lua:multithread(nThread)
   if self._train_set then
      self._train_set:multithread(nThread)
   end
   if self._valid_set then
      self._valid_set:multithread(nThread)
   end
e
