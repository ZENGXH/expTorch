require 'nn'
local opt = {}

--[[ solver ]]--
opt.learning_rate = 0.01 --starting learning rate
opt.learningRate_decay = 0.000001 -- opt.learning_rate * 0.05 
opt.weight_decay_iter = 12000
opt.batch_size = 200 --number of sequences to train on in parallel
opt.train_epoch_size = 2000
opt.momentum = 0.9
opt.model_train_multiselect = true
opt.random_select = 1

--[[ test and save ]]--
opt.model_save_path = '.'
opt.save_acc_thre = 0.9
opt.save_iter = 500
opt.test_interval = 100 * 20 
opt.test_iter = 20*10

--[[ stragegy ]]--
opt.inputLength = 1 
opt.unitPixels = opt.inputLength/2 - 1
opt.glimpseLength = 1
opt.transfer = 'ReLU'-- 'activation function')

--[[ reinforce ]]--
opt.rewardScale = 1 -- "scale of positive reward (negative is 0)")
opt.locatorStd = 0.15 -- 'stdev of gaussian location sampler (between 0 and 1) (low values may cause NaNs)')
opt.stochastic = false -- 'Reinforce modules forward inputs stochastically during evaluation')

--[[ glimpse layer ]]--
-- in glimpse layer, input shape: B, (D)feature_size, (H)frames_per_select, (W)feature_h*feature_w
opt.glimpseHiddenSize = 2 -- 'size of glimpse hidden layer')
opt.glimpseScale_h = 2 -- 'scale of successive patches w.r.t. original input image')
opt.glimpseScale_w = 1
opt.glimpseDepth = 1 -- 'number of concatenated downscaled patches')
opt.locatorHiddenSize = 8 -- 'size of locator hidden layer')
opt.imageHiddenSize = 101 -- 'size of hidden layer combining glimpse and locator hiddens')

--[[ recurrent layer ]]--
opt.rho = 1 -- 'back-propagate through time (BPTT) for rho time-steps')
opt.hiddenSize = 8 -- 'number of hidden units used in Simple RNN.')
opt.FastLSTM = true --true-- 'use LSTM instead of linear layer')
opt.recurrent = nn.Identity() 

--[[ parameters for self design ]]--
local Model = require 'model.video_model'
opt.model_function = Model.GlimpseH1
opt.finetune_model_parameters = 'para_epoch4'

-- optimization
opt.cont_epoch = 1 --
if opt.finetune_model_parameters then
    opt.snapshot_prefix =  'cl_c_'..opt.finetune_model_parameters  --'train_batch256'
    opt.finetune_model_parameters = opt.finetune_model_parameters..'.t7'
else
    opt.snapshot_prefix =  'para'  --'train_batch256'
end
opt.log_name = opt.snapshot_prefix..'.log'

--[[ general ]]--
opt.uniform = 0.003 -- 'initialize parameters using uniform distribution between -uniform and uniform. -1 means default initialization')
opt.progress = true-- 'print progress bar')
opt.silent = false-- 'dont print anything to stdout')
opt.nThread = 1
opt.seed = 1
opt.verbose=true
opt.cache_mode='overwrite'

--[[ display ]]--
opt.loss_display_iter = 50
opt.cont_iter = 0
opt.max_iterations = 30000
--[[ dataSource ]]--
opt.feature_size = 1024
opt.feature_h = 1
opt.feature_w = 1 
opt.parameters_init = 0.003
opt.num_selection = 1 
opt.load_size={1, 1024, 1, 1}
opt.sample_size={1024, 1, 1}
opt.input_type='flow'

opt.classID_file='/data1/wangjiang/datasets/ucf101_list/ClassID_ucf101.txt'
opt.data_train_list='/data1/zengxiaohui/experiment/action_data/ucf101_list/train_list_split1.txt'
opt.data_test_list='/data1/zengxiaohui/experiment/action_data/ucf101_list/test_list_split1.txt'
opt.data_root='/data1/zengxiaohui/experiment/action_data/ucf101'
opt.data_folder='seg5_video'
opt.frames_per_select_train=1

-- GPU/CPU
opt.cuda = true -- false --true -- true -- 'use CUDA')
opt.time = 0 --print batch times
opt.cudnn = 1 --use cudnn (1=yes). this should greatly speed up convolutions
opt.device_id = 2

--[[ task info ]]--
opt.num_class = 101
return opt
