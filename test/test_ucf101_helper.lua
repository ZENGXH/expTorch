package.path = package.path..";/data1/zengxiaohui/dprnn/dprnn.lua"
dp = dofile '../dprnn.lua'
helper = dofile '../utils/ucf101_helper.lua'
mode = 'train'
root = '/data1/zengxiaohui/experiment/action_data/ucf101/ucf101_flow/ucf101_flow_'..mode
seg5_bin_path = 'seg5'

reduce_length = 10
input_file = '/data1/zengxiaohui/experiment/action_data/ucf101_list/'..mode..'_vidlist_flow_split1.txt'
helper.MvFile(input_file, root, reduce_length, seg5_bin_path)
