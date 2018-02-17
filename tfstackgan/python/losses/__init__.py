"""Exposes a color loss used in StackGAN."""
from tensorflow.python.util.all_util import remove_undocumented

from tfstackgan.python.losses.python import losses_impl
from tfstackgan.python.losses.python.losses_impl import *

_allowed_symbols = []
_allowed_symbols += losses_impl.__all__
remove_undocumented(__name__, _allowed_symbols)
