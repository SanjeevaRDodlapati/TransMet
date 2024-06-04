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
        self.scope = 'dna'
    def inputs(self, dna_wlen):
        return [kl.Input(shape=(dna_wlen, 4), name='dna')]


def bs(x, df=None, knots=None, degree=3, intercept=False):
    """
    df : int
        The number of degrees of freedom to use for this spline. The
        return value will have this many columns. You must specify at least
        one of `df` and `knots`.
    knots : list(float)
        The interior knots of the spline. If unspecified, then equally
        spaced quantiles of the input data are used. You must specify at least
        one of `df` and `knots`.
    degree : int
        The degree of the piecewise polynomial. Default is 3 for cubic splines.
    intercept : bool
        If `True`, the resulting spline basis will span the intercept term
        (i.e. the constant function). If `False` (the default) then this
        will not be the case, which is useful for avoiding overspecification
        in models that include multiple spline terms and/or an intercept term.

    """

    order = degree + 1
    inner_knots = []
    if df is not None and knots is None:
        n_inner_knots = df - order + (1 - intercept)
        if n_inner_knots < 0:
            n_inner_knots = 0
            print("df was too small; have used %d"
                  % (order - (1 - intercept)))

        if n_inner_knots > 0:
            inner_knots = np.percentile(
                x, 100 * np.linspace(0, 1, n_inner_knots + 2)[1:-1])

    elif knots is not None:
        inner_knots = knots

    all_knots = np.concatenate(
        ([np.min(x), np.max(x)] * order, inner_knots))

    all_knots.sort()

    n_basis = len(all_knots) - (degree + 1)
    basis = np.empty((x.shape[0], n_basis), dtype=float)

    for i in range(n_basis):
        coefs = np.zeros((n_basis,))
        coefs[i] = 1
        basis[:, i] = splev(x, (all_knots, coefs, degree))

    if not intercept:
        basis = basis[:, 1:]
    return basis


def spline_factory(n, df, log=False):
    if log:
        dist = np.array(np.arange(n) - n / 2.0)
        dist = np.log(np.abs(dist) + 1) * (2 * (dist > 0) - 1)
        n_knots = df - 4
        knots = np.linspace(np.min(dist), np.max(dist), n_knots + 2)[1:-1]
        return tf.convert_to_tensor(bs(
            dist, knots=knots, intercept=True))
    else:
        dist = np.arange(n)
        return tf.convert_to_tensor(bs(
            dist, df=df, intercept=True))


class BSplineTransformation(tf.keras.layers.Layer):
    def __init__(self, degrees_of_freedom, log=False, scaled=False):
        super(BSplineTransformation, self).__init__()
        self._spline_tr = None
        self._log = log
        self._scaled = scaled
        self._df = degrees_of_freedom
        self.spatial_dim = 960

    def __call__(self, input):
        if self._spline_tr is None:
            # spatial_dim = input.size()[-1]
            self._spline_tr = spline_factory(self.spatial_dim, self._df, log=self._log)
            if self._scaled:
                self._spline_tr = self._spline_tr / self.spatial_dim
            # if input.is_cuda:
            # self._spline_tr = self._spline_tr.cuda()

        return tf.matmul(input, self._spline_tr)


class BSplineConv1D(DnaModel):

    def __init__(self, in_channels, out_channels, kernel_size, degrees_of_freedom, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, log=False, scaled=True):
        super(BSplineConv1D, self).__init__()
        self._df = degrees_of_freedom
        self._log = log
        self._scaled = scaled

        self.spline = kl.Conv1d(filters=degrees_of_freedom,
                                kernel_size=kernel_size,
                                strides=stride,
                                padding=padding,
                                dialation_rate=dilation,
                                use_bias=False)
        self.spline.weight = spline_factory(kernel_size, self._df, log=log).view(self._df, 1, kernel_size)
        if scaled:
            self.spline.weight = self.spline.weight / kernel_size
        self.spline.weight = tf.Variable(self.spline.weight)
        self.spline.weight.requires_grad = False
        self.conv1d = kl.Conv1d(out_channels, 1, groups=groups, use_bias=bias)

    def forward(self, input):
        batch_size, n_channels, length = input.size()
        spline_out = self.spline(input.view(batch_size * n_channels, 1, length))
        conv1d_out = self.conv1d(spline_out.view(batch_size, n_channels * self._df, length))
        return conv1d_out


class Sei(DnaModel):
    def __init__(self, nb_hidden=72, *args, **kwargs):
        super(Sei, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden
        self.df = 16
        self.spline_tr = BSplineTransformation(degrees_of_freedom=self.df, scaled=False)

    def __call__(self, inputs):
        x = inputs[0]

        # kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        lout1 = kl.Conv1D(filters=480, kernel_size=9, padding='same')(x)
        lout1 = kl.Conv1D(filters=480, kernel_size=9, padding='same')(lout1)

        out1 = kl.Conv1D(filters=480, kernel_size=9, padding='same')(lout1)
        out1 = kl.Activation('relu')(out1)
        out1 = kl.Conv1D(filters=480, kernel_size=9, padding='same')(out1)
        out1 = kl.Activation('relu')(out1)

        lout2 = kl.MaxPooling1D(4)(out1 + lout1)
        lout2 = kl.Dropout(0.2)(lout2)
        lout2 = kl.Conv1D(filters=640, kernel_size=9, padding='same')(lout2)
        lout2 = kl.Conv1D(filters=640, kernel_size=9, padding='same')(lout2)

        out2 = kl.Dropout(0.2)(lout2)
        out2 = kl.Conv1D(filters=640, kernel_size=9, padding='same')(out2)
        out2 = kl.Activation('relu')(out2)
        out2 = kl.Conv1D(filters=640, kernel_size=9, padding='same')(out2)
        out2 = kl.Activation('relu')(out2)

        lout3 = kl.MaxPooling1D(4)(out2 + lout2)
        lout3 = kl.Dropout(0.2)(lout3)
        lout3 = kl.Conv1D(filters=960, kernel_size=9, padding='same')(lout3)
        lout3 = kl.Conv1D(filters=960, kernel_size=9, padding='same')(lout3)

        out3 = kl.Dropout(0.2)(lout3)
        out3 = kl.Conv1D(filters=960, kernel_size=9, padding='same')(out3)
        out3 = kl.Activation('relu')(out3)
        out3 = kl.Conv1D(filters=960, kernel_size=9, padding='same')(out3)
        out3 = kl.Activation('relu')(out3)

        deconv1 = kl.Dropout(0.10)(out3 + lout3)
        deconv1 = kl.Conv1D(filters=960, kernel_size=5, dilation_rate=2, padding='same')(deconv1)
        deconv1 = kl.Activation('relu')(deconv1)

        cat1 = out3 + deconv1
        deconv2 = kl.Dropout(0.10)(cat1)
        deconv2 = kl.Conv1D(filters=960, kernel_size=5, dilation_rate=4, padding='same')(deconv2)
        deconv2 = kl.Activation('relu')(deconv2)

        cat2 = cat1 + deconv2
        deconv3 = kl.Dropout(0.10)(cat2)
        deconv3 = kl.Conv1D(filters=960, kernel_size=5, dilation_rate=8, padding='same')(deconv3)
        deconv3 = kl.Activation('relu')(deconv3)

        cat3 = cat2 + deconv3
        deconv4 = kl.Dropout(0.10)(cat3)
        deconv4 = kl.Conv1D(filters=960, kernel_size=5, dilation_rate=16, padding='same')(deconv4)
        deconv4 = kl.Activation('relu')(deconv4)

        cat4 = cat3 + deconv4
        deconv5 = kl.Dropout(0.10)(cat4)
        deconv5 = kl.Conv1D(filters=960, kernel_size=5, dilation_rate=25, padding='same')(deconv5)
        deconv5 = kl.Activation('relu')(deconv5)
        out = cat4 + deconv5

        out = kl.Dropout(0.5)(out)

        out = self.spline_tr(out)

        # x = kl.Flatten()(x)
        out = kl.Dense(self.nb_hidden)(out)
        # x = kl.Dense(512)(x)
        out = kl.Activation('relu')(out)

        return self._build(inputs, out)


class DeepSEA(DnaModel):
    def __init__(self, nb_hidden=512, *args, **kwargs):
        super(DeepSEA, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]

        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(320, 8, kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)
        x = kl.Dropout(0.2)(x)

        x = kl.Conv1D(480, 8, kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)
        x = kl.Dropout(0.2)(x)

        x = kl.Conv1D(960, 8, kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(0.5)(x)

        x = kl.Flatten()(x)

        x = kl.Dense(self.nb_hidden, kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)

        return self._build(inputs, x)


class DeeperDeepSEA(DnaModel):
    def __init__(self, nb_hidden=512, *args, **kwargs):
        super(DeeperDeepSEA, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]

        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(320, 8, kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Conv1D(320, 8, kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)
        x = kl.BatchNormalization()(x)

        x = kl.Conv1D(480, 8, kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Conv1D(480, 8, kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)
        x = kl.BatchNormalization()(x)
        x = kl.Dropout(0.2)(x)

        x = kl.Conv1D(960, 8, kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Conv1D(960, 8, kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.BatchNormalization()(x)
        x = kl.Dropout(0.2)(x)

        x = kl.Flatten()(x)

        x = kl.Dense(self.nb_hidden, kernel_regularizer=kernel_regularizer)(x)
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

class CnnL2h32(DnaModel):
    """CNN with two convolutional and one fully-connected layer with 128 units.

    .. code::

        Parameters: 4,100,000
        Specification: conv[128@11]_mp[4]_conv[256@3]_mp[2]_fc[128]_do
    """

    def __init__(self, nb_hidden=32, *args, **kwargs):
        super(CnnL2h32, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]

        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(32, 11,
                      kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)
        if self.batch_norm:
            x = kl.BatchNormalization()(x)

        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(64, 3,
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


class CnnL2h64(DnaModel):
    """CNN with two convolutional and one fully-connected layer with 128 units.

    .. code::

        Parameters: 4,100,000
        Specification: conv[128@11]_mp[4]_conv[256@3]_mp[2]_fc[128]_do
    """

    def __init__(self, nb_hidden=64, *args, **kwargs):
        super(CnnL2h64, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]

        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(64, 11,
                      kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)
        if self.batch_norm:
            x = kl.BatchNormalization()(x)

        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(128, 3,
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

class CnnL3h32(DnaModel):
    """CNN with three convolutional and one fully-connected layer with 128 units.

    .. code::

        Parameters: 4,400,000
        Specification: conv[128@11]_mp[4]_conv[256@3]_mp[2]_conv[512@3]_mp[2]_
                       fc[128]_do
    """

    def __init__(self, nb_hidden=32, *args, **kwargs):
        super(CnnL3h32, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]

        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(32, 11,
                      kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)

        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(64, 3,
                      kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(2)(x)

        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(128, 3,
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


class CnnL3h64(DnaModel):
    """CNN with three convolutional and one fully-connected layer with 128 units.

    .. code::

        Parameters: 4,400,000
        Specification: conv[128@11]_mp[4]_conv[256@3]_mp[2]_conv[512@3]_mp[2]_
                       fc[128]_do
    """

    def __init__(self, nb_hidden=64, *args, **kwargs):
        super(CnnL3h64, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]

        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(64, 11,
                      kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)

        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(128, 3,
                      kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(2)(x)

        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(256, 3,
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


class CnnRnn01(DnaModel):
    """Convolutional-recurrent model.

    Convolutional-recurrent model with two convolutional layers followed by a
    bidirectional GRU layer.

    .. code::

        Parameters: 1,100,000
        Specification: conv[128@11]_pool[4]_conv[256@7]_pool[4]_bgru[256]_do
    """

    def __call__(self, inputs):
        x = inputs[0]

        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(128, 11,
                      kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)

        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(256, 7,
                      kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)

        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        gru = kl.recurrent.GRU(256, kernel_regularizer=kernel_regularizer)
        x = kl.Bidirectional(gru)(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)


class ResNet01(DnaModel):
    """Residual network with bottleneck residual units.

    .. code::

        Parameters: 1,700,000
        Specification: conv[128@11]_mp[2]_resb[2x128|2x256|2x512|1x1024]_gap_do

    He et al., 'Identity Mappings in Deep Residual Networks.'
    """

    def _res_unit(self, inputs, nb_filter, size=3, stride=1, stage=1, block=1):

        name = '%02d-%02d/' % (stage, block)
        id_name = '%sid_' % (name)
        res_name = '%sres_' % (name)

        # Residual branch

        # 1x1 down-sample conv
        x = kl.BatchNormalization(name=res_name + 'bn1')(inputs)
        x = kl.Activation('relu', name=res_name + 'act1')(x)
        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(nb_filter[0], 1,
                      name=res_name + 'conv1',
                      subsample_length=stride,
                      kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)

        # LxL conv
        x = kl.BatchNormalization(name=res_name + 'bn2')(x)
        x = kl.Activation('relu', name=res_name + 'act2')(x)
        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(nb_filter[1], size,
                      name=res_name + 'conv2',
                      border_mode='same',
                      kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)

        # 1x1 up-sample conv
        x = kl.BatchNormalization(name=res_name + 'bn3')(x)
        x = kl.Activation('relu', name=res_name + 'act3')(x)
        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(nb_filter[2], 1,
                      name=res_name + 'conv3',
                      kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)

        # Identity branch
        if nb_filter[-1] != inputs._keras_shape[-1] or stride > 1:
            kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
            identity = kl.Conv1D(nb_filter[2], 1,
                                 name=id_name + 'conv1',
                                 subsample_length=stride,
                                 kernel_initializer=self.init,
                                 kernel_regularizer=kernel_regularizer)(inputs)
        else:
            identity = inputs

        x = kl.merge([identity, x], name=name + 'merge', mode='sum')

        return x

    def __call__(self, inputs):
        x = inputs[0]

        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(128, 11,
                      name='conv1',
                      kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.BatchNormalization(name='bn1')(x)
        x = kl.Activation('relu', name='act1')(x)
        x = kl.MaxPooling1D(2, name='pool1')(x)

        # 124
        x = self._res_unit(x, [32, 32, 128], stage=1, block=1, stride=2)
        x = self._res_unit(x, [32, 32, 128], stage=1, block=2)

        # 64
        x = self._res_unit(x, [64, 64, 256], stage=2, block=1, stride=2)
        x = self._res_unit(x, [64, 64, 256], stage=2, block=2)

        # 32
        x = self._res_unit(x, [128, 128, 512], stage=3, block=1, stride=2)
        x = self._res_unit(x, [128, 128, 512], stage=3, block=2)

        # 16
        x = self._res_unit(x, [256, 256, 1024], stage=4, block=1, stride=2)

        x = kl.GlobalAveragePooling1D()(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)


class ResNet02(ResNet01):
    """Residual network with bottleneck residual units.

    .. code::

        Parameters: 2,000,000
        Specification: conv[128@11]_mp[2]_resb[3x128|3x256|3x512|1x1024]_gap_do

    He et al., 'Identity Mappings in Deep Residual Networks.'
    """

    def __call__(self, inputs):
        x = inputs[0]

        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(128, 11,
                      name='conv1',
                      kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.BatchNormalization(name='bn1')(x)
        x = kl.Activation('relu', name='act1')(x)
        x = kl.MaxPooling1D(2, name='pool1')(x)

        # 124
        x = self._res_unit(x, [32, 32, 128], stage=1, block=1, stride=2)
        x = self._res_unit(x, [32, 32, 128], stage=1, block=2)
        x = self._res_unit(x, [32, 32, 128], stage=1, block=3)

        # 64
        x = self._res_unit(x, [64, 64, 256], stage=2, block=1, stride=2)
        x = self._res_unit(x, [64, 64, 256], stage=2, block=2)
        x = self._res_unit(x, [64, 64, 256], stage=2, block=3)

        # 32
        x = self._res_unit(x, [128, 128, 512], stage=3, block=1, stride=2)
        x = self._res_unit(x, [128, 128, 512], stage=3, block=2)
        x = self._res_unit(x, [128, 128, 512], stage=3, block=3)

        # 16
        x = self._res_unit(x, [256, 256, 1024], stage=4, block=1, stride=2)

        x = kl.GlobalAveragePooling1D()(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)


class ResConv01(ResNet01):
    """Residual network with two convolutional layers in each residual unit.

    .. code::

        Parameters: 2,800,000
        Specification: conv[128@11]_mp[2]_resc[2x128|1x256|1x256|1x512]_gap_do

    He et al., 'Identity Mappings in Deep Residual Networks.'
    """

    def _res_unit(self, inputs, nb_filter, size=3, stride=1, stage=1, block=1):

        name = '%02d-%02d/' % (stage, block)
        id_name = '%sid_' % (name)
        res_name = '%sres_' % (name)

        # Residual branch
        x = kl.BatchNormalization(name=res_name + 'bn1')(inputs)
        x = kl.Activation('relu', name=res_name + 'act1')(x)
        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(nb_filter, size,
                      name=res_name + 'conv1',
                      border_mode='same',
                      subsample_length=stride,
                      kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)

        x = kl.BatchNormalization(name=res_name + 'bn2')(x)
        x = kl.Activation('relu', name=res_name + 'act2')(x)
        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(nb_filter, size,
                      name=res_name + 'conv2',
                      border_mode='same',
                      kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)

        # Identity branch
        if nb_filter != inputs._keras_shape[-1] or stride > 1:
            kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
            identity = kl.Conv1D(nb_filter, size,
                                 name=id_name + 'conv1',
                                 border_mode='same',
                                 subsample_length=stride,
                                 kernel_initializer=self.init,
                                 kernel_regularizer=kernel_regularizer)(inputs)
        else:
            identity = inputs

        x = kl.merge([identity, x], name=name + 'merge', mode='sum')

        return x

    def __call__(self, inputs):
        x = inputs[0]

        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(128, 11,
                      name='conv1',
                      kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.BatchNormalization(name='bn1')(x)
        x = kl.Activation('relu', name='act1')(x)
        x = kl.MaxPooling1D(2, name='pool1')(x)

        # 124
        x = self._res_unit(x, 128, stage=1, block=1, stride=2)
        x = self._res_unit(x, 128, stage=1, block=2)

        # 64
        x = self._res_unit(x, 256, stage=2, block=1, stride=2)

        # 32
        x = self._res_unit(x, 256, stage=3, block=1, stride=2)

        # 32
        x = self._res_unit(x, 512, stage=4, block=1, stride=2)

        x = kl.GlobalAveragePooling1D()(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)


class ResAtrous01(DnaModel):
    """Residual network with Atrous (dilated) convolutional layers.

    Residual network with Atrous (dilated) convolutional layer in bottleneck
    units. Atrous convolutional layers allow to increase the receptive field and
    hence better model long-range dependencies.

    .. code::

        Parameters: 2,000,000
        Specification: conv[128@11]_mp[2]_resa[3x128|3x256|3x512|1x1024]_gap_do

    He et al., 'Identity Mappings in Deep Residual Networks.'
    Yu and Koltun, 'Multi-Scale Context Aggregation by Dilated Convolutions.'
    """

    def _res_unit(self, inputs, nb_filter, size=3, stride=1, atrous=1,
                  stage=1, block=1):

        name = '%02d-%02d/' % (stage, block)
        id_name = '%sid_' % (name)
        res_name = '%sres_' % (name)

        # Residual branch

        # 1x1 down-sample conv
        x = kl.BatchNormalization(name=res_name + 'bn1')(inputs)
        x = kl.Activation('relu', name=res_name + 'act1')(x)
        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(nb_filter[0], 1,
                      name=res_name + 'conv1',
                      subsample_length=stride,
                      kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)

        # LxL conv
        x = kl.BatchNormalization(name=res_name + 'bn2')(x)
        x = kl.Activation('relu', name=res_name + 'act2')(x)
        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(nb_filter[1], size,
                            dilation_rate=atrous,
                            name=res_name + 'conv2',
                            border_mode='same',
                            kernel_initializer=self.init,
                            kernel_regularizer=kernel_regularizer)(x)
#         x = kl.AtrousConv1D(nb_filter[1], size,
#                             atrous_rate=atrous,
#                             name=res_name + 'conv2',
#                             border_mode='same',
#                             kernel_initializer=self.init,
#                             kernel_regularizer=kernel_regularizer)(x)

        # 1x1 up-sample conv
        x = kl.BatchNormalization(name=res_name + 'bn3')(x)
        x = kl.Activation('relu', name=res_name + 'act3')(x)
        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(nb_filter[2], 1,
                      name=res_name + 'conv3',
                      kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)

        # Identity branch
        if nb_filter[-1] != inputs._keras_shape[-1] or stride > 1:
            kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
            identity = kl.Conv1D(nb_filter[2], 1,
                                 name=id_name + 'conv1',
                                 subsample_length=stride,
                                 kernel_initializer=self.init,
                                 kernel_regularizer=kernel_regularizer)(inputs)
        else:
            identity = inputs

        x = kl.merge([identity, x], name=name + 'merge', mode='sum')

        return x

    def __call__(self, inputs):
        x = inputs[0]

        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(128, 11,
                      name='conv1',
                      kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu', name='act1')(x)
        x = kl.MaxPooling1D(2, name='pool1')(x)

        # 124
        x = self._res_unit(x, [32, 32, 128], stage=1, block=1, stride=2)
        x = self._res_unit(x, [32, 32, 128], atrous=2, stage=1, block=2)
        x = self._res_unit(x, [32, 32, 128], atrous=4, stage=1, block=3)

        # 64
        x = self._res_unit(x, [64, 64, 256], stage=2, block=1, stride=2)
        x = self._res_unit(x, [64, 64, 256], atrous=2, stage=2, block=2)
        x = self._res_unit(x, [64, 64, 256], atrous=4, stage=2, block=3)

        # 32
        x = self._res_unit(x, [128, 128, 512], stage=3, block=1, stride=2)
        x = self._res_unit(x, [128, 128, 512], atrous=2, stage=3, block=2)
        x = self._res_unit(x, [128, 128, 512], atrous=4, stage=3, block=3)

        # 16
        x = self._res_unit(x, [256, 256, 1024], stage=4, block=1, stride=2)

        x = kl.GlobalAveragePooling1D()(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)


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
