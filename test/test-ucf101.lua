-- package.path = package.path..";/data1/zengxiaohui/dprnn/dprnn.lua;/data1/zengxiaohui/dprnn/utils/?.lua"

require 'dprnn'

dp.DATA_DIR='/data1/zengxiaohui/experiment/action_data/ucf101'
local ds = dp.UCF101{
    load_size={1, 1024, 1, 1},
    sample_size={1024, 1, 1},
    input_type='flow',
    -- for videoclassset
    classID_file='/data1/wangjiang/datasets/ucf101_list/ClassID_ucf101.txt',
    data_list='/data1/zengxiaohui/experiment/action_data/ucf101_list/test_list_split1.txt',
    data_root='/data1/zengxiaohui/experiment/action_data/ucf101',
    data_folder='seg5_video',
    sample_func='sampleDefault',
    which_set='test',
    frames_per_select=1,
    verbose=true,
    cache_mode='writeonce',
    }

-- ds:multithread(1)
-- local ds = dp.ucf101(config)

testSet = ds:loadTest()
print('load done @')
batch = testSet:sample(1)
assert(batch:inputs())
assert(batch:targets())
print('sample done @')
-- ds:get('test', 'targets')
