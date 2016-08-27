require 'nnx'
require 'dprnnnn'
require 'string'
_ = require 'moses'
require 'xlua'
require 'os'
require 'sys'
require 'image'
require 'lfs'
require 'torchx'
ffi = require 'ffi'

------------------------------------------------------------------------
--[[ dprnn ]]--
-- deep learning library for torch7.
------------------------------------------------------------------------

dprnn = {}
dprnn.TORCH_DIR = os.getenv('TORCH_DATA_PATH') or os.getenv('HOME')
   
--[[ utils ]]--
torch.include('dprnn', 'utils/utils.lua')
torch.include('dprnn', 'utils/underscore.lua')
torch.include('dprnn', 'utils/os.lua')
torch.include('dprnn', 'utils/table.lua')
torch.include('dprnn', 'utils/torch.lua')

--[[ directory structure ]]--
dprnn.DATA_DIR = os.getenv('DEEP_DATA_PATH') 
   or paths.concat(dprnn.TORCH_DIR, 'data')
dprnn.mkdir(dp.DATA_DIR)

dprnn.SAVE_DIR = os.getenv('DEEP_SAVE_PATH') 
   or paths.concat(dprnn.TORCH_DIR, 'save')
dprnn.mkdir(dp.SAVE_DIR)

dprnn.LOG_DIR = os.getenv('DEEP_LOG_PATH') 
   or paths.concat(dprnn.TORCH_DIR, 'log')
dprnn.mkdir(dp.LOG_DIR)

dprnn.UNIT_DIR = os.getenv('DEEP_UNIT_PATH') 
   or paths.concat(dprnn.TORCH_DIR, 'unit')
dprnn.mkdir(dp.UNIT_DIR)
   
--[[ misc ]]--
torch.include('dprnn', 'xplog.lua')
torch.include('dprnn', 'mediator.lua')
torch.include('dprnn', 'objectid.lua')

--[[ view ]]--
torch.include('dprnn', 'view/view.lua')
torch.include('dprnn', 'view/dataview.lua')
torch.include('dprnn', 'view/imageview.lua')
torch.include('dprnn', 'view/classview.lua')
torch.include('dprnn', 'view/sequenceview.lua')
torch.include('dprnn', 'view/listview.lua')

--[[ dataset ]]--
-- datasets
torch.include('dprnn', 'data/baseset.lua') -- abstract class
torch.include('dprnn', 'data/dataset.lua')
torch.include('dprnn', 'data/sentenceset.lua')
torch.include('dprnn', 'data/textset.lua')
torch.include('dprnn', 'data/imageclassset.lua')
torch.include('dprnn', 'data/batch.lua')

--[[ datasource ]]--
-- generic datasources
torch.include('dprnn', 'data/datasource.lua')
torch.include('dprnn', 'data/imagesource.lua')
torch.include('dprnn', 'data/smallimagesource.lua')
torch.include('dprnn', 'data/textsource.lua')
-- specific image datasources
torch.include('dprnn', 'data/mnist.lua')
torch.include('dprnn', 'data/cifar10.lua')
torch.include('dprnn', 'data/cifar100.lua')
torch.include('dprnn', 'data/notmnist.lua')
torch.include('dprnn', 'data/facialkeypoints.lua')
torch.include('dprnn', 'data/svhn.lua')
torch.include('dprnn', 'data/imagenet.lua')
torch.include('dprnn', 'data/facedetection.lua')
torch.include('dprnn', 'data/translatedmnist.lua')
-- specific text datasources
torch.include('dprnn', 'data/billionwords.lua')
torch.include('dprnn', 'data/penntreebank.lua')

--[[ sampler ]]--
torch.include('dprnn', 'sampler/sampler.lua')
torch.include('dprnn', 'sampler/shufflesampler.lua')
torch.include('dprnn', 'sampler/sentencesampler.lua')
torch.include('dprnn', 'sampler/randomsampler.lua')
torch.include('dprnn', 'sampler/textsampler.lua')

--[[ preprocess ]]--
torch.include('dprnn', 'preprocess/preprocess.lua')
torch.include('dprnn', 'preprocess/pipeline.lua')
torch.include('dprnn', 'preprocess/parallelpreprocess.lua')
torch.include('dprnn', 'preprocess/binarize.lua')
torch.include('dprnn', 'preprocess/standardize.lua')
torch.include('dprnn', 'preprocess/gcn.lua')
torch.include('dprnn', 'preprocess/zca.lua')
torch.include('dprnn', 'preprocess/lecunlcn.lua')

--[[ propagator ]]--
torch.include('dprnn', 'propagator/propagator.lua')
torch.include('dprnn', 'propagator/optimizer.lua')
torch.include('dprnn', 'propagator/evaluator.lua')
torch.include('dprnn', 'propagator/experiment.lua')

--[[ feedback ]]--
torch.include('dprnn', 'feedback/feedback.lua')
torch.include('dprnn', 'feedback/compositefeedback.lua')
torch.include('dprnn', 'feedback/confusion.lua')
torch.include('dprnn', 'feedback/perplexity.lua')
torch.include('dprnn', 'feedback/topcrop.lua')
torch.include('dprnn', 'feedback/fkdkaggle.lua')
torch.include('dprnn', 'feedback/facialkeypointfeedback.lua')

--[[ observer ]]--
torch.include('dprnn', 'observer/observer.lua')
torch.include('dprnn', 'observer/compositeobserver.lua')
torch.include('dprnn', 'observer/logger.lua')
torch.include('dprnn', 'observer/errorminima.lua')
torch.include('dprnn', 'observer/earlystopper.lua')
torch.include('dprnn', 'observer/savetofile.lua') --not an observer (but used in one)
torch.include('dprnn', 'observer/adaptivedecay.lua')
torch.include('dprnn', 'observer/filelogger.lua')
torch.include('dprnn', 'observer/hyperlog.lua')

--[[ nn ]]--
torch.include('dprnn', 'nn/Print.lua')
torch.include('dprnn', 'nn/FairLookupTable.lua')

--[[ test ]]--
torch.include('dprnn', 'test/test.lua')
torch.include('dprnn', 'test/test-cuda.lua')
torch.include('dprnn', 'test/test-datasets.lua')

return dprnn
