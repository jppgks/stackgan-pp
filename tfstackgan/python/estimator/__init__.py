"""Exposes Estimator tools used in StackGAN."""
from tensorflow.python.util.all_util import remove_undocumented

from tfstackgan.python.estimator.python import gan_estimator_impl
from tfstackgan.python.estimator.python import head_impl
from tfstackgan.python.estimator.python.gan_estimator_impl import *
from tfstackgan.python.estimator.python.head_impl import *

_allowed_symbols = []
_allowed_symbols += gan_estimator_impl.__all__
_allowed_symbols += head_impl.__all__
remove_undocumented(__name__, _allowed_symbols)
