
require 'nnx'
require 'dpnn'
require 'string'
_ = require 'moses'
require 'xlua'
require 'os'
require 'sys'
require 'nn'
require 'image'
require 'lfs'
require 'torchx'
ffi = require 'ffi'

------------------------------------------------------------------------
--[[ dp ]]--
-- deep learning library for torch7.
------------------------------------------------------------------------

dp = {}
dp.TORCH_DIR = os.getenv('TORCH_DATA_PATH') or os.getenv('HOME')
dp.EXP_DIR = os.getenv('PWD')   
dp.DPRNN_DIR='/data1/zengxiaohui/dprnn'
--[[ utils ]]--
dofile(paths.concat(dp.DPRNN_DIR, 'utils/utils.lua'))
dofile(paths.concat(dp.DPRNN_DIR, 'utils/underscore.lua'))
dofile(paths.concat(dp.DPRNN_DIR, 'utils/os.lua'))
dofile(paths.concat(dp.DPRNN_DIR, 'utils/table.lua'))
dofile(paths.concat(dp.DPRNN_DIR, 'utils/torch.lua'))
dofile(paths.concat(dp.DPRNN_DIR, 'utils/ucf101_helper.lua'))
dofile(paths.concat(dp.DPRNN_DIR, 'utils/helper.lua'))
--[[ directory structure ]]--
dp.DATA_DIR = os.getenv('DEEP_DATA_PATH') 
   or paths.concat(dp.TORCH_DIR, 'data')
dp.mkdir(dp.DATA_DIR)

dp.SAVE_DIR = os.getenv('DEEP_SAVE_PATH') 
   or paths.concat(dp.TORCH_DIR, 'save')
dp.mkdir(dp.SAVE_DIR)

dp.LOG_DIR = os.getenv('DEEP_LOG_PATH') 
   or paths.concat(dp.TORCH_DIR, 'log')
dp.mkdir(dp.LOG_DIR)

dp.UNIT_DIR = os.getenv('DEEP_UNIT_PATH') 
   or paths.concat(dp.TORCH_DIR, 'unit')
dp.mkdir(dp.UNIT_DIR)
   
--[[ misc ]]--
dofile(paths.concat(dp.DPRNN_DIR, 'xplog.lua'))
dofile(paths.concat(dp.DPRNN_DIR, 'mediator.lua'))
dofile(paths.concat(dp.DPRNN_DIR, 'objectid.lua'))

--[[ view ]]--
dofile(paths.concat(dp.DPRNN_DIR, 'view/view.lua'))
dofile(paths.concat(dp.DPRNN_DIR, 'view/dataview.lua'))
dofile(paths.concat(dp.DPRNN_DIR, 'view/imageview.lua'))
dofile(paths.concat(dp.DPRNN_DIR, 'view/classview.lua'))
dofile(paths.concat(dp.DPRNN_DIR, 'view/sequenceview.lua'))
dofile(paths.concat(dp.DPRNN_DIR, 'view/listview.lua'))
dofile(paths.concat(dp.DPRNN_DIR, 'view/videoview.lua'))

--[[ dataset ]]--
-- datasets
dofile(paths.concat(dp.DPRNN_DIR, 'data/baseset.lua')) -- abstract class
dofile(paths.concat(dp.DPRNN_DIR, 'data/dataset.lua'))
dofile(paths.concat(dp.DPRNN_DIR, 'data/sentenceset.lua'))
dofile(paths.concat(dp.DPRNN_DIR, 'data/textset.lua'))
dofile(paths.concat(dp.DPRNN_DIR, 'data/imageclassset.lua'))
dofile(paths.concat(dp.DPRNN_DIR, 'data/batch.lua'))
dofile(paths.concat(dp.DPRNN_DIR, 'data/videoclassset.lua'))
--[[ datasource ]]--
-- generic datasources
dofile(paths.concat(dp.DPRNN_DIR, 'data/datasource.lua'))
dofile(paths.concat(dp.DPRNN_DIR, 'data/imagesource.lua'))
dofile(paths.concat(dp.DPRNN_DIR, 'data/smallimagesource.lua'))
dofile(paths.concat(dp.DPRNN_DIR, 'data/textsource.lua'))
-- specific image datasources
dofile(paths.concat(dp.DPRNN_DIR, 'data/ucf101.lua'))
dofile(paths.concat(dp.DPRNN_DIR, 'data/mnist.lua'))
dofile(paths.concat(dp.DPRNN_DIR, 'data/cifar10.lua'))
dofile(paths.concat(dp.DPRNN_DIR, 'data/cifar100.lua'))
dofile(paths.concat(dp.DPRNN_DIR, 'data/notmnist.lua'))
dofile(paths.concat(dp.DPRNN_DIR, 'data/facialkeypoints.lua'))
dofile(paths.concat(dp.DPRNN_DIR, 'data/svhn.lua'))
dofile(paths.concat(dp.DPRNN_DIR, 'data/imagenet.lua'))
dofile(paths.concat(dp.DPRNN_DIR, 'data/facedetection.lua'))
dofile(paths.concat(dp.DPRNN_DIR, 'data/translatedmnist.lua'))
-- specific text datasources
dofile(paths.concat(dp.DPRNN_DIR, 'data/billionwords.lua'))
dofile(paths.concat(dp.DPRNN_DIR, 'data/penntreebank.lua'))

--[[ sampler ]]--
dofile(paths.concat(dp.DPRNN_DIR, 'sampler/sampler.lua'))
dofile(paths.concat(dp.DPRNN_DIR, 'sampler/shufflesampler.lua'))
dofile(paths.concat(dp.DPRNN_DIR, 'sampler/sentencesampler.lua'))
dofile(paths.concat(dp.DPRNN_DIR, 'sampler/randomsampler.lua'))
dofile(paths.concat(dp.DPRNN_DIR, 'sampler/textsampler.lua'))

--[[ preprocess ]]--
dofile(paths.concat(dp.DPRNN_DIR, 'preprocess/preprocess.lua'))
dofile(paths.concat(dp.DPRNN_DIR, 'preprocess/pipeline.lua'))
dofile(paths.concat(dp.DPRNN_DIR, 'preprocess/parallelpreprocess.lua'))
dofile(paths.concat(dp.DPRNN_DIR, 'preprocess/binarize.lua'))
dofile(paths.concat(dp.DPRNN_DIR, 'preprocess/standardize.lua'))
dofile(paths.concat(dp.DPRNN_DIR, 'preprocess/gcn.lua'))
dofile(paths.concat(dp.DPRNN_DIR, 'preprocess/zca.lua'))
dofile(paths.concat(dp.DPRNN_DIR, 'preprocess/lecunlcn.lua'))

--[[ propagator ]]--
dofile(paths.concat(dp.DPRNN_DIR, 'propagator/propagator.lua'))
dofile(paths.concat(dp.DPRNN_DIR, 'propagator/optimizer.lua'))
dofile(paths.concat(dp.DPRNN_DIR, 'propagator/evaluator.lua'))
dofile(paths.concat(dp.DPRNN_DIR, 'propagator/experiment.lua'))

--[[ feedback ]]--
dofile(paths.concat(dp.DPRNN_DIR, 'feedback/feedback.lua'))
dofile(paths.concat(dp.DPRNN_DIR, 'feedback/compositefeedback.lua'))
dofile(paths.concat(dp.DPRNN_DIR, 'feedback/confusion.lua'))
dofile(paths.concat(dp.DPRNN_DIR, 'feedback/perplexity.lua'))
dofile(paths.concat(dp.DPRNN_DIR, 'feedback/topcrop.lua'))
dofile(paths.concat(dp.DPRNN_DIR, 'feedback/fkdkaggle.lua'))
dofile(paths.concat(dp.DPRNN_DIR, 'feedback/facialkeypointfeedback.lua'))

--[[ observer ]]--
dofile(paths.concat(dp.DPRNN_DIR, 'observer/observer.lua'))
dofile(paths.concat(dp.DPRNN_DIR, 'observer/compositeobserver.lua'))
dofile(paths.concat(dp.DPRNN_DIR, 'observer/logger.lua'))
dofile(paths.concat(dp.DPRNN_DIR, 'observer/errorminima.lua'))
dofile(paths.concat(dp.DPRNN_DIR, 'observer/earlystopper.lua'))
dofile(paths.concat(dp.DPRNN_DIR, 'observer/savetofile.lua')) --not an observer (but used in one)
dofile(paths.concat(dp.DPRNN_DIR, 'observer/adaptivedecay.lua'))
dofile(paths.concat(dp.DPRNN_DIR, 'observer/filelogger.lua'))
dofile(paths.concat(dp.DPRNN_DIR, 'observer/hyperlog.lua'))

--[[ nn ]]--
dofile(paths.concat(dp.DPRNN_DIR, 'nn/Print.lua'))
dofile(paths.concat(dp.DPRNN_DIR, 'nn/FairLookupTable.lua'))

--[[ test ]]--
dofile(paths.concat(dp.DPRNN_DIR, 'test/test.lua'))
dofile(paths.concat(dp.DPRNN_DIR, 'test/test-cuda.lua'))
dofile(paths.concat(dp.DPRNN_DIR, 'test/test-datasets.lua'))

return dp
