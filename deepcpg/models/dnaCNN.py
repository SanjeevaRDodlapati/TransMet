"""DNA models.

Provides models trained with DNA sequence windows.
"""

from __future__ import division
from __future__ import print_function

import inspect

import tensorflow as tf
from tensorflow.keras import layers as kl
from tensorflow.keras import regularizers as kr

from .utils import Model
from ..utils import get_from_module

import numpy as np
from scipy.interpolate import splev


class DnaModel(Model):
    """Abstract class of a DNA model."""
    def __init__(self, *args, **kwargs):
        super(DnaModel, self).__init__(*args, **kwargs)
        self.scope = 'dnaCNN'
    def inputs(self, dna_wlen):
        return [kl.Input(shape=(dna_wlen, 4), name='dna')]


class DeepSEA(DnaModel):
    def __init__(self, nb_hidden=512, nb_features=None, *args, **kwargs):
        super(DeepSEA, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden
        self.nb_features = nb_features

    def __call__(self, inputs):
        x = inputs[0]

        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(320, 8, activation='relu', kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.MaxPooling1D(4)(x)
        x = kl.Dropout(0.2)(x)

        x = kl.Conv1D(480, 8, activation='relu', kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.MaxPooling1D(4)(x)
        x = kl.Dropout(0.2)(x)

        x = kl.Conv1D(960, 8, activation='relu', kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Dropout(0.5)(x)

        x = kl.Flatten()(x)

        x = kl.Dense(self.nb_hidden, activation='relu', kernel_initializer=self.init,
                     kernel_regularizer=kernel_regularizer)(x)
        x = kl.Dense(self.nb_features, activation='sigmoid', kernel_initializer=self.init,
                     kernel_regularizer=kernel_regularizer)(x)

        return self._build(inputs, x)


class DeeperDeepSEA(DnaModel):
    def __init__(self, nb_hidden=512, *args, **kwargs):
        super(DeeperDeepSEA, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]

        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(320, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Conv1D(320, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)
        x = kl.BatchNormalization()(x)

        x = kl.Conv1D(480, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Conv1D(480, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)
        x = kl.BatchNormalization()(x)
        x = kl.Dropout(0.2)(x)

        x = kl.Conv1D(960, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Conv1D(960, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.BatchNormalization()(x)
        x = kl.Dropout(0.2)(x)

        x = kl.Flatten()(x)

        x = kl.Dense(self.nb_hidden, kernel_initializer=self.init,
                     kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.BatchNormalization()(x)

        return self._build(inputs, x)


class CnnL1h128(DnaModel):
    """CNN with one convolutional and one fully-connected layer with 128 units.

    .. code::

        Parameters: 4,100,000
        Specification: conv[128@11]_mp[4]_fc[128]_do
    """

    def __init__(self, nb_hidden=128, *args, **kwargs):
        super(CnnL1h128, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]

        kernel_regularizer = kr.L1L2(self.l1_decay, self.l2_decay)
        x = kl.Conv1D(128, 11,
                      kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)

        x = kl.Flatten()(x)

        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(self.nb_hidden,
                     kernel_initializer=self.init,
                     kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)


class CnnL1h256(CnnL1h128):
    """CNN with one convolutional and one fully-connected layer with 256 units.

    .. code::

        Parameters: 8,100,000
        Specification: conv[128@11]_mp[4]_fc[256]_do
    """

    def __init__(self,  *args, **kwargs):
        super(CnnL1h256, self).__init__(*args, **kwargs)
        self.nb_hidden = 256


class CnnL2h128(DnaModel):
    """CNN with two convolutional and one fully-connected layer with 128 units.

    .. code::

        Parameters: 4,100,000
        Specification: conv[128@11]_mp[4]_conv[256@3]_mp[2]_fc[128]_do
    """

    def __init__(self, nb_hidden=128, *args, **kwargs):
        super(CnnL2h128, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]

        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(128, 11,
                      kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)
        if self.batch_norm:
            x = kl.BatchNormalization()(x)

        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(256, 3,
                      kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(2)(x)
        if self.batch_norm:
            x = kl.BatchNormalization()(x)

        x = kl.Flatten()(x)

        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(self.nb_hidden,
                     kernel_initializer=self.init,
                     kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)

class CnnL2h256(CnnL2h128):
    """CNN with two convolutional and one fully-connected layer with 256 units.

    .. code::

        Parameters: 8,100,000
        Specification: conv[128@11]_mp[4]_conv[256@3]_mp[2]_fc[256]_do
    """

    def __init__(self,  *args, **kwargs):
        super(CnnL2h256, self).__init__(*args, **kwargs)
        self.nb_hidden = 256


class CnnL3h128(DnaModel):
    """CNN with three convolutional and one fully-connected layer with 128 units.

    .. code::

        Parameters: 4,400,000
        Specification: conv[128@11]_mp[4]_conv[256@3]_mp[2]_conv[512@3]_mp[2]_
                       fc[128]_do
    """

    def __init__(self, nb_hidden=128, *args, **kwargs):
        super(CnnL3h128, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]

        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(128, 11,
                      kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)

        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(256, 3,
                      kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(2)(x)

        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(512, 3,
                      kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(2)(x)

        x = kl.Flatten()(x)

        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(self.nb_hidden,
                     kernel_initializer=self.init,
                     kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)


class CnnL3h256(CnnL3h128):
    """CNN with three convolutional and one fully-connected layer with 256 units.

    .. code::

        Parameters: 8,300,000
        Specification: conv[128@11]_mp[4]_conv[256@3]_mp[2]_conv[512@3]_mp[2]_
                       fc[256]_do
    """

    def __init__(self,  *args, **kwargs):
        super(CnnL3h256, self).__init__(*args, **kwargs)
        self.nb_hidden = 256



def list_models():
    """Return the name of models in the module."""

    models = dict()
    for name, value in globals().items():
        if inspect.isclass(value) and name.lower().find('model') == -1:
            models[name] = value
    return models


def get(name):
    """Return object from module by its name."""
    return get_from_module(name, globals())
