import sys
CAFFE_ROOT = '/home/karan/Desktop/Sem-3/15-769/caffe/build/'
sys.path.insert(0, CAFFE_ROOT + 'python')
import caffe
caffe.set_mode_cpu()

NUM_DATA = {}
NUM_DATA['TRAIN'] = 10000
NUM_DATA['VAL'] = 1000
NUM_DATA['TEST'] = 1000

RAND_SEED = {}
RAND_SEED['TRAIN'] = 19654
RAND_SEED['VAL'] = 2353
RAND_SEED['TEST'] = 51235
