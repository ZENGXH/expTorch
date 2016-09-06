require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'
require 'dprnn.dprnn'
local cmd = torch.CmdLine()
cmd:option('-c','opt_1024.lua','initial')
local args = cmd:parse(arg)
print('get args: ', args.c)
local optfile = args.c 
assert(optfile)
log = loadfile(paths.concat(dp.DPRNN_DIR, 'utils', 'log.lua'))()
log.SetLoggerName('main')
log.info('loading optfile: ', optfile)

local opt = dofile(optfile)
torch.setnumthreads(opt.nThread)
log.info('loading optfile done ')
log.outfile = opt.log_name or 'train.log'
torch.manualSeed(opt.seed)
if opt.cuda then
    require 'cutorch'
    require 'cudnn'
    require 'cunn'
    cutorch.setDevice(opt.device_id)
 end


--[[ criterion ]]--
dp.DATA_DIR='/data1/zengxiaohui/experiment/action_data/ucf101'
assert(opt.classID_file)
assert(opt.frames_per_select_train)
local ds = dp.UCF101{
    load_all = true,
    load_size=opt.load_size, -- {1, 1024, 1, 1},
    sample_size=opt.sample_size, --{1024, 1, 1},
    input_type=opt.input_type, --'flow',
    -- for videoclassset
    classID_file=opt.classID_file, --'/data1/wangjiang/datasets/ucf101_list/ClassID_ucf101.txt',
    data_test_list=opt.data_test_list,
    data_train_list=opt.data_train_list,
    -- data_list=opt.data_list, --'/data1/zengxiaohui/experiment/action_data/ucf101_list/test_list_split1.txt',
    data_root=opt.data_root, --'/data1/zengxiaohui/experiment/action_data/ucf101',
    data_folder=opt.data_folder, --'seg5_video',
    sample_func=opt.sample_func, --'sampleDefault',
    which_set=opt.which_set, --'test',
    verbose=opt.verbose, --true,    
    frames_per_select_train=opt.frames_per_select_train,--1,
    cache_mode=opt.cache_mode --'writeonce',
}
local Model_dict = dofile('model/video_model.lua') 
model = Model_dict.GlimpseH1(opt, ds)
print"Model :"
print(model)
 model.agent = agent
-- torch.save('model.t7', model)
-- 
-- model = torch.load('model.t7')
assert(model and torch.isTypeOf(model, 'nn.Module'))

--[[ criterion ]]--
-- criterion for backward
local VRC_module = nn.ModuleCriterion(nn.VRClassReward(model, opt.rewardScale), nil, nil)
local Entropy_module = nn.ModuleCriterion(nn.ClassNLLCriterion(), nn.Log(), nil)
local criterion = nn.ParallelCriterion(true)
    :add(Entropy_module) -- BACKPROP
    :add(VRC_module) -- REINFORCE


assert(ds and ds ~= nil)

-- criterion for loss display and accuracy
local criterion_show = nn.ModuleCriterion(nn.ClassNLLCriterion(), nn.Log())
if opt.cuda then
end
--[[ feedback ]]--
-- two component, first one which select the first output 
-- second one use Confusion matrix
local feedback_conf = dp.Confusion{
    name='classication_acc',
    bce=false, 
    target_dim=-1, 
    output_module=nn.Sequential():add(nn.SelectTable(1))
}
local feedback_rego_loss = dp.Criteria({
    name='classication_loss',
    criteria={criterion_show},
    output_to_report=true,
    output_module = {nn.SelectTable(1)},
})
local composeFeedback = dp.CompositeFeedback{
    name='compose_feedback',
    feedbacks={feedback_rego_loss, feedback_conf}
}

--[[ ppf ]]--
opt.ppf = function(batch) --#TODO only for ucf101 , moved into it
   --[[
    local inputView = batch:GetView('input')
    assert(inputView:view() == 'btchw')
    local input = inputView:GetInputTensor()
    inputView:forwardPut('btcw', input:select(4, 1))
    batch:SetView('input', inputView)
    ]]--
    return batch
end
--[[ Propagators ]]--
local PrinterObserverConfig = {
        name='train_printer',
        channels='printer',
        display_interval=20
    }

--[ train Propagators ]--
train = dp.Optimizer{
    observer=dp.PrinterObserver(PrinterObserverConfig),
    stats = true,
    verbose = true,
    acc_update = opt.accUpdate,
    loss = criterion,
    callback = function(model, report) 
        opt.learning_rate =  opt.learning_rate
        if opt.accUpdate then
            model:accUpdateGradParameters(model.dpnn_input, 
                model.output, opt.learning_rate)
        else
            model:updateGradParameters(opt.momentum) -- affects gradParams
            -- model:weightDecay(opt.weightDecay) --affects gradParams
            model:updateParameters(opt.learning_rate) -- affects params
        end
        -- model:maxParamNorm(opt.maxOutNorm) -- affects params
        model:zeroGradParameters() -- affects gradParams 
    end,
    sampler = dp.InorderSampler{
        batch_size=opt.batch_size, 
        epoch_size=opt.train_epoch_size,
        ppf=opt.ppf
    },
    feedback = composeFeedback,
    progress = opt.progress
}
--[ valid Propagators ]--
PrinterObserverConfig.name='test_printer'
test = dp.Evaluator{
   observer=dp.PrinterObserver(PrinterObserverConfig),
    stats = true,
    verbose = true,
    feedback = composeFeedback, 
    sampler = dp.InorderSampler{
        batch_size=opt.batch_size, 
        epoch_size=opt.train_epoch_size,
        ppf=opt.ppf
    }
}

--[[ model 
]]--
--[[multithreading]]--
if opt.nThread > 0 then
    log.info('setting multithread as ', opt.nThread, 
    ' with batch_size', opt.batch_size)
    -- dp.VisualDataSet.multithread(ds, opt.nThread, opt.batch_size) --opt.batch_size)
    ds:multithread(opt.nThread, opt.batch_size) --opt.batch_size)

    train:sampler():async()
    test:sampler():async()
else
    log.info('not using multithread')
end


--[[Experiment
]]--
xp = dp.Experiment{
    stats = true,
    verbose = true,
    model = model,
    optimizer = train,
    tester = test,
    --[[
    observer = {
        dp.FileLogger(),
        dp.EarlyStopper{
            error_report = {'validator', 'feedback', 'classication_loss'},
            maximize = true,
            max_epochs = opt.maxTries
        }
    },
    ]]--
    random_seed = os.time(),
    max_epoch = opt.maxEpoch
}

--[[GPU or CPU]]--
if opt.cuda then
    log.info('cuda mode')
    log.info('setDevice: ', opt.device_id)
    log.info('setting cuda')
    xp:cuda()
end


--[[
RandomSampler:setup({})
sampler = RandomSampler:sampleEpoch(ds)
local batch

local timer = torch.Timer()
local time = timer:time().real
for i = 1, 50 do
batch, i, n = sampler(batch)
input = batch:inputs():input()
target = batch:targets():input()
end

print("load 10 batch time", timer:time().real - time)
assert(input)
assert(target)
]]--
xp:run(ds)

-- vim:ts=3 ss=3 sw=3 expandtab
