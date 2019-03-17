# -*- coding: utf-8 -*-
from .utils import cost
from .utils import accuracy
from .utils import xavier_initializer, xavier_initializer_wb
from .utils import relu, sigmoid, tanh, softsign, identity, sat, clu, softplus, elu, arctan
from .utils import fc_layer, softmax_layer, srn_layer
from .utils import fetch_mnist, view_mnist
from .utils import string2onehot_matrix, onehot_matrix2string
from .utils import txt2data, one_hot_vector
from .utils import make_dict, build_dataset, one_hot_vector
from .utils import log_loss, binary_log_loss
from .utils import cost, accuracy, log_loss, binary_log_loss
from .xerion import Xerion


__version__ = '0.3'
__author__ = 'Shin Asakawa'
__license__ = 'Apache License, Version 2.0'
__email__ = 'asakawa@ieee.org'
__copyright__ = 'Copyright 2018 2019 {0}'.format(__author__)
