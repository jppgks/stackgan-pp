"""TFGAN framework tailored to StackGAN."""
from tensorflow.python.util.all_util import remove_undocumented

from tfstackgan.python import losses
from tfstackgan.python import namedtuples
from tfstackgan.python import train
from tfstackgan.python import estimator
from tfstackgan.python.namedtuples import *
from tfstackgan.python.train import *

_allowed_symbols = ['losses', 'estimator']
_allowed_symbols += train.__all__
_allowed_symbols += namedtuples.__all__
remove_undocumented(__name__, _allowed_symbols)
