"""DNA models.

Provides models trained with DNA sequence windows.
"""

from __future__ import division
from __future__ import print_function

import inspect

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as kl
from tensorflow.keras import regularizers as kr
from scipy.interpolate import splev

# import sonnet as snt

from .utils import Model
from ..utils import get_from_module



    
def gelu(x: tf.Tensor) -> tf.Tensor:
    return tf.nn.sigmoid(1.702 * x) * x


class DnaModel(Model):
    """Abstract class of a DNA model."""
    def __init__(self, *args, **kwargs):
        super(DnaModel, self).__init__(*args, **kwargs)
        self.scope = 'dna'
    def inputs(self, dna_wlen):
        return [kl.Input(shape=(dna_wlen, 4), name='dna')]



class ChromBPNet(DnaModel):
    def __init__(self, nb_hidden=512, *args, **kwargs):
        super(ChromBPNet, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden
        
        # default parameters
        self.conv1_kernel_size = 21
        self.profile_kernel_size = 75
        # self.num_tasks = 1 
        self.n_dil_layers = 3 #e example 
        

    def __call__(self, inputs):
        x = inputs[0]

        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        filter = 64

        # first convolution without dilation
        x = kl.Conv1D(filters=filter,
                      kernel_size=self.conv1_kernel_size,
                      padding='valid')(x)
        x = kl.Activation('relu')(x)

        # dilation layers
        layer_names = [str(i) for i in range(1, self.n_dil_layers + 1)]
        for i in range(1, self.n_dil_layers + 1):
            conv_layer_name = f'bpnet_{layer_names[i-1]}conv'
            conv_x = kl.Conv1D(filters=filter,
                               kernel_size=3,
                               padding='valid',
                               dilation_rate=2**i)(x)
            conv_x = kl.Activation('relu')(conv_x)
            x_len = K.int_shape(x)[1]
            conv_x_len = K.int_shape(conv_x)[1]
            assert((x_len - conv_x_len) % 2 == 0)
            
            x = kl.Cropping1D((x_len - conv_x_len) // 2, name=f"bpnet_{layer_names[i-1]}crop")(x)
            x = kl.add([conv_x, x])

        # x = kl.Flatten()(x)
        x = kl.GlobalAvgPool1D()(x)

        # fully connected layer
        # x = kl.Dense(self.nb_hidden, kernel_initializer=self.init,
        #             kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)


        # fully connecteed layer for single output per sequence
        x = kl.Dense(1)(x)

        return self._build(inputs, x)


class DeepSEA(DnaModel):
    def __init__(self, nb_hidden=512, *args, **kwargs):
        super(DeepSEA, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]

        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(320, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)
        x = kl.Dropout(0.2)(x)

        x = kl.Conv1D(480, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)
        x = kl.Dropout(0.2)(x)

        x = kl.Conv1D(960, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        # x = kl.MaxPooling1D(3)(x)
        x = kl.Dropout(0.5)(x)

        x = kl.Flatten()(x)

        x = kl.Dense(self.nb_hidden, kernel_initializer=self.init,
                     kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)

        return self._build(inputs, x)




class AdvancedDilatedDNA(DnaModel):
    """
        AdvancedDilatedDNA is a deep model for one-hot encoded DNA sequences 
    that incorporates residual connections, batch normalization, and dilated convolutions
    to DeepSEA as a base model architecture.
    
    """

    def __init__(self, nb_hidden=512, *args, **kwargs):
        super(AdvancedDilatedDNA, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]
        
        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)

        ## Block 1 (No pooling, residual connection)
        shortcut1 = kl.Conv1D(320, 1, padding='same', kernel_initializer=self.init,
                              kernel_regularizer=kernel_regularizer)(x)
        x = kl.Conv1D(320, 8, padding='same', kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Conv1D(320, 8, dilation_rate=2, padding='same', kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Add()([x, shortcut1])
        x = kl.Dropout(0.2)(x)

        ## Block 2 (With pooling, residual connection)
        shortcut2 = kl.Conv1D(480, 1, padding='same', kernel_initializer=self.init,
                              kernel_regularizer=kernel_regularizer)(kl.MaxPooling1D(4)(x))
        x = kl.Conv1D(480, 8, padding='same', kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)
        x = kl.Conv1D(480, 8, dilation_rate=4, padding='same', kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Add()([x, shortcut2])
        x = kl.Dropout(0.2)(x)

        ## Block 3 (With pooling, residual connection)
        shortcut3 = kl.Conv1D(640, 1, padding='same', kernel_initializer=self.init,
                              kernel_regularizer=kernel_regularizer)(kl.MaxPooling1D(4)(x))
        x = kl.Conv1D(640, 8, padding='same', kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)
        x = kl.Conv1D(640, 8, dilation_rate=8, padding='same', kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Add()([x, shortcut3])
        x = kl.Dropout(0.3)(x)

        # Final Stage
        x = kl.GlobalAveragePooling1D()(x)
        x = kl.Dense(self.nb_hidden, kernel_initializer=self.init,
                     kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)

        return self._build(inputs, x)




class LongDilDNA(DnaModel):
    """
    LongDilDNA is designed to capture long-range interactions
    from very long one-hot encoded DNA sequence inputs (e.g., 16k or 32k in length).
    
    Using "same" padding in all convolutional layers, the sequence length is maintained 
    (except for downsampling from pooling). With 5 blocks and MaxPooling1D with pool_size=4
    in each block, the overall downsampling factor is 4^5 = 1024.
    
    For example:
      - For a 32k input, final sequence width ≈ 32000/1024 ≈ 31 positions.
      - For a 16k input, final sequence width ≈ 16000/1024 ≈ 16 positions.
    
    Architecture:
      Block 1:
         - Conv1D: 320 filters, kernel size=8, dilation_rate=1, padding="same"
         - BatchNormalization, ReLU
         - MaxPooling1D with pool_size=4
         - Conv1D: 320 filters, kernel size=8, dilation_rate=2, padding="same"
         - BatchNormalization, ReLU
         - Dropout (0.2)
      
      Block 2:
         - Conv1D: 480 filters, kernel size=8, dilation_rate=1, padding="same"
         - BatchNormalization, ReLU
         - MaxPooling1D with pool_size=4
         - Conv1D: 480 filters, kernel size=8, dilation_rate=4, padding="same"
         - BatchNormalization, ReLU
         - Dropout (0.2)
      
      Block 3:
         - Conv1D: 640 filters, kernel size=8, dilation_rate=1, padding="same"
         - BatchNormalization, ReLU
         - MaxPooling1D with pool_size=4
         - Conv1D: 640 filters, kernel size=8, dilation_rate=8, padding="same"
         - BatchNormalization, ReLU
         - Dropout (0.3)
      
      Block 4:
         - Conv1D: 960 filters, kernel size=8, dilation_rate=1, padding="same"
         - BatchNormalization, ReLU
         - MaxPooling1D with pool_size=4
         - Conv1D: 960 filters, kernel size=8, dilation_rate=8, padding="same"
         - BatchNormalization, ReLU
         - Dropout (0.5)
      
      Block 5:
         - Conv1D: 1024 filters, kernel size=8, dilation_rate=1, padding="same"
         - BatchNormalization, ReLU
         - MaxPooling1D with pool_size=4
         - Conv1D: 1024 filters, kernel size=8, dilation_rate=8, padding="same"
         - BatchNormalization, ReLU
         - Dropout (0.5)
      
      Final Stage:
         - GlobalAveragePooling1D
         - Dense(nb_hidden) with BatchNormalization and ReLU
         - Output is produced via self._build(inputs, x)
    """
    def __init__(self, nb_hidden=512, *args, **kwargs):
        super(LongDilDNA, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]
        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        
        # Block 1
        x = kl.Conv1D(320, 8, dilation_rate=1, padding='same', kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.BatchNormalization()(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)
        x = kl.Conv1D(320, 8, dilation_rate=2, padding='same', kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.BatchNormalization()(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(0.2)(x)
        
        # Block 2
        x = kl.Conv1D(480, 8, dilation_rate=1, padding='same', kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.BatchNormalization()(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)
        x = kl.Conv1D(480, 8, dilation_rate=4, padding='same', kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.BatchNormalization()(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(0.2)(x)
        
        # Block 3
        x = kl.Conv1D(640, 8, dilation_rate=1, padding='same', kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.BatchNormalization()(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)
        x = kl.Conv1D(640, 8, dilation_rate=8, padding='same', kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.BatchNormalization()(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(0.3)(x)
        
        # Block 4
        x = kl.Conv1D(960, 8, dilation_rate=1, padding='same', kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.BatchNormalization()(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)
        x = kl.Conv1D(960, 8, dilation_rate=8, padding='same', kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.BatchNormalization()(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(0.5)(x)
        
        # Block 5
        x = kl.Conv1D(1024, 8, dilation_rate=1, padding='same', kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.BatchNormalization()(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)
        x = kl.Conv1D(1024, 8, dilation_rate=8, padding='same', kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.BatchNormalization()(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(0.5)(x)
        
        # Final Stage
        x = kl.GlobalAveragePooling1D()(x)
        x = kl.Dense(self.nb_hidden, kernel_initializer=self.init,
                     kernel_regularizer=kernel_regularizer)(x)
        x = kl.BatchNormalization()(x)
        x = kl.Activation('relu')(x)
        
        return self._build(inputs, x)



class DeepDilatedDNA(DnaModel):
    def __init__(self, nb_hidden=512, *args, **kwargs):
        super(DeepDilatedDNA, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]
        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        
        # Block 1: Regular conv -> pooling -> dropout -> dilated conv (kernel size 8, dilation rate 2)
        x = kl.Conv1D(320, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)
        # Dilated conv with kernel size 8, dilation rate 2
        x = kl.Conv1D(320, 8, dilation_rate=2, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(0.2)(x)
        
        # Block 2: Regular conv -> pooling -> dropout -> dilated conv (kernel size 8, dilation rate 4)
        x = kl.Conv1D(480, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)
        # Dilated conv with kernel size 8, dilation rate 4
        x = kl.Conv1D(480, 8, dilation_rate=4, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(0.2)(x)
        
        # Block 3: Regular conv -> dropout -> dilated conv (kernel size 8, dilation rate 8)
        x = kl.Conv1D(960, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        # Dilated conv with kernel size 8, dilation rate 8
        x = kl.Conv1D(960, 8, dilation_rate=8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(0.5)(x)
        
        # Flatten and fully connected layers
        x = kl.Flatten()(x)
        x = kl.Dense(self.nb_hidden, kernel_initializer=self.init,
                     kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        
        return self._build(inputs, x)




class DilatedDNAFreeze(DnaModel):
    """
    A DeepSEA-style model with an extra convolution block and dilated convolutions.
    Unlike a doubling strategy, after reaching a dilation rate of 4, the dilation
    is frozen at 4 in subsequent blocks.
    
    Architecture:
      Block 1: Regular Conv (320 filters, kernel=8) → MaxPool (stride=4) → Dropout
               → Dilated Conv (320 filters, kernel=8, dilation_rate=2)
      Block 2: Regular Conv (528 filters, kernel=8) → MaxPool (stride=4) → Dropout
               → Dilated Conv (528 filters, kernel=8, dilation_rate=4)
      Block 3: Regular Conv (736 filters, kernel=8) → MaxPool (stride=4) → Dropout
               → Dilated Conv (736 filters, kernel=8, dilation_rate=4)  [Frozen]
      Block 4: Regular Conv (960 filters, kernel=8) → Dropout
               → Dilated Conv (960 filters, kernel=8, dilation_rate=4)  [Frozen]
      Followed by Flatten and Dense layers.
    """
    def __init__(self, nb_hidden=512, *args, **kwargs):
        super(DilatedDNAFreeze, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]
        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        
        # Block 1: 320 filters, dilation rate = 2
        x = kl.Conv1D(320, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)
        x = kl.Conv1D(320, 8, dilation_rate=2, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(0.2)(x)
        
        # Block 2: 528 filters, dilation rate = 4
        x = kl.Conv1D(528, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)
        x = kl.Conv1D(528, 8, dilation_rate=4, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(0.3)(x)
        
        # Block 3: 736 filters, pooling followed by dilated conv with frozen dilation rate = 4
        x = kl.Conv1D(736, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)
        x = kl.Conv1D(736, 8, dilation_rate=4, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(0.4)(x)
        
        # Block 4: 960 filters, no pooling; dilated conv with frozen dilation rate = 4
        x = kl.Conv1D(960, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Conv1D(960, 8, dilation_rate=4, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(0.5)(x)
        
        # Flatten and fully connected layer to produce final features.
        x = kl.Flatten()(x)
        x = kl.Dense(self.nb_hidden, kernel_initializer=self.init,
                     kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        
        return self._build(inputs, x)
        
        
        
class DilatedL4(DnaModel):
    """
    A DeepSEA-style model with an extra convolution block and dilated convolutions.
    Unlike a doubling strategy, after reaching a dilation rate of 4, the dilation
    is frozen at 4 in subsequent blocks.
    
    Architecture:
      Block 1: Regular Conv (320 filters, kernel=8) → MaxPool (stride=4) → Dropout
               → Dilated Conv (320 filters, kernel=8, dilation_rate=2)
      Block 2: Regular Conv (528 filters, kernel=8) → MaxPool (stride=4) → Dropout
               → Dilated Conv (528 filters, kernel=8, dilation_rate=4)
      Block 3: Regular Conv (736 filters, kernel=8) → MaxPool (stride=4) → Dropout
               → Dilated Conv (736 filters, kernel=8, dilation_rate=4)  [Frozen]
      Block 4: Regular Conv (960 filters, kernel=8) → Dropout
               → Dilated Conv (960 filters, kernel=8, dilation_rate=4)  [Frozen]
      Followed by Flatten and Dense layers.
    """
    def __init__(self, nb_hidden=512, *args, **kwargs):
        super(DilatedL4, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]
        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        
        # Block 1: 320 filters, dilation rate = 2
        x = kl.Conv1D(320, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(2)(x)
        x = kl.Conv1D(320, 8, dilation_rate=2, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(0.2)(x)
        
        # Block 2: 528 filters, dilation rate = 4
        x = kl.Conv1D(528, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)
        x = kl.Conv1D(528, 8, dilation_rate=4, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(0.3)(x)
        
        # Block 3: 736 filters, pooling followed by dilated conv with frozen dilation rate = 4
        x = kl.Conv1D(736, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)
        x = kl.Conv1D(736, 8, dilation_rate=8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(0.4)(x)
        
        # Block 4: 960 filters, no pooling; dilated conv with frozen dilation rate = 4
        x = kl.Conv1D(960, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Conv1D(960, 8, dilation_rate=16, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(0.5)(x)
        
        # Flatten and fully connected layer to produce final features.
        x = kl.Flatten()(x)
        x = kl.Dense(self.nb_hidden, kernel_initializer=self.init,
                     kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        
        return self._build(inputs, x)
        
        
        
class DilatedL4Alt(DnaModel):
    """
    A DeepSEA-style model with an extra convolution block and dilated convolutions.
    Unlike a doubling strategy, after reaching a dilation rate of 4, the dilation
    is frozen at 4 in subsequent blocks.
    
    Architecture:
      Block 1: Regular Conv (320 filters, kernel=8) → MaxPool (stride=4) → Dropout
               → Dilated Conv (320 filters, kernel=8, dilation_rate=2)
      Block 2: Regular Conv (528 filters, kernel=8) → MaxPool (stride=4) → Dropout
               → Dilated Conv (528 filters, kernel=8, dilation_rate=4)
      Block 3: Regular Conv (736 filters, kernel=8) → MaxPool (stride=4) → Dropout
               → Dilated Conv (736 filters, kernel=8, dilation_rate=4)  [Frozen]
      Block 4: Regular Conv (960 filters, kernel=8) → Dropout
               → Dilated Conv (960 filters, kernel=8, dilation_rate=4)  [Frozen]
      Followed by Flatten and Dense layers.
    """
    def __init__(self, nb_hidden=512, *args, **kwargs):
        super(DilatedL4Alt, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]
        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        
        # Block 1: 320 filters, dilation rate = 2
        x = kl.Conv1D(320, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        # x = kl.MaxPooling1D(2)(x)
        x = kl.Conv1D(320, 8, dilation_rate=2, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(0.2)(x)
        
        # Block 2: 528 filters, dilation rate = 4
        x = kl.Conv1D(528, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)
        x = kl.Conv1D(528, 8, dilation_rate=4, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(0.3)(x)
        
        # Block 3: 736 filters, pooling followed by dilated conv with frozen dilation rate = 4
        x = kl.Conv1D(736, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)
        x = kl.Conv1D(736, 8, dilation_rate=4, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(0.4)(x)
        
        # Block 4: 960 filters, no pooling; dilated conv with frozen dilation rate = 4
        x = kl.Conv1D(960, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)
        x = kl.Conv1D(960, 8, dilation_rate=4, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(0.5)(x)
        
        # Flatten and fully connected layer to produce final features.
        x = kl.Flatten()(x)
        x = kl.Dense(self.nb_hidden, kernel_initializer=self.init,
                     kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        
        return self._build(inputs, x)
        
        
        
        
class Dilated6L(DnaModel):
    def __init__(self, nb_hidden=512, *args, **kwargs):
        super(Dilated6L, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]
        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        
        # Block 1: 128 filters, no pooling, dilation_rate = 2, dropout 0.1
        x = kl.Conv1D(128, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Conv1D(128, 8, dilation_rate=2, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(0.1)(x)
        
        # Block 2: 256 filters, max pool size = 2, dilation_rate = 4, dropout 0.2
        x = kl.Conv1D(256, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(2)(x)
        x = kl.Conv1D(256, 8, dilation_rate=4, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(0.2)(x)
        
        # Block 3: 384 filters, max pool size = 2, dilation_rate = 4, dropout 0.2
        x = kl.Conv1D(384, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(2)(x)
        x = kl.Conv1D(384, 8, dilation_rate=4, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(0.2)(x)
        
        # Block 4: 512 filters, max pool size = 2, dilation_rate = 4, dropout 0.3
        x = kl.Conv1D(512, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(3)(x)
        x = kl.Conv1D(512, 8, dilation_rate=4, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(0.3)(x)
        
        # Block 5: 640 filters, max pool size = 3, dilation_rate = 4, dropout 0.4
        x = kl.Conv1D(640, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(3)(x)
        x = kl.Conv1D(640, 8, dilation_rate=4, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(0.4)(x)
        
        # Block 6: 768 filters, max pool size = 4, dilation_rate = 4, dropout 0.5
        x = kl.Conv1D(768, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(3)(x)
        x = kl.Conv1D(768, 8, dilation_rate=4, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(0.5)(x)
        
        # Flatten and fully connected layer to produce final features.
        x = kl.Flatten()(x)
        x = kl.Dense(self.nb_hidden, kernel_initializer=self.init,
                     kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        
        return self._build(inputs, x)
        
        

class Dilated6Light(DnaModel):
    def __init__(self, nb_hidden=512, *args, **kwargs):
        super(Dilated6Light, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]
        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        
        # Block 1: 64 filters, no pooling, dilation_rate = 2, dropout 0.1
        x = kl.Conv1D(64, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Conv1D(64, 8, dilation_rate=2, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(0.1)(x)
        
        # Block 2: 128 filters, max pool size = 2, dilation_rate = 4, dropout 0.2
        x = kl.Conv1D(128, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(2)(x)
        x = kl.Conv1D(128, 8, dilation_rate=4, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(0.2)(x)
        
        # Block 3: 192 filters, max pool size = 2, dilation_rate = 4, dropout 0.2
        x = kl.Conv1D(192, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(2)(x)
        x = kl.Conv1D(192, 8, dilation_rate=4, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(0.2)(x)
        
        # Block 4: 256 filters, max pool size = 3, dilation_rate = 4, dropout 0.3
        x = kl.Conv1D(256, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(3)(x)
        x = kl.Conv1D(256, 8, dilation_rate=4, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(0.3)(x)
        
        # Block 5: 320 filters, max pool size = 3, dilation_rate = 4, dropout 0.4
        x = kl.Conv1D(320, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(3)(x)
        x = kl.Conv1D(320, 8, dilation_rate=4, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(0.4)(x)
        
        # Block 6: 384 filters, max pool size = 3, dilation_rate = 4, dropout 0.5
        x = kl.Conv1D(384, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(3)(x)
        x = kl.Conv1D(384, 8, dilation_rate=4, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(0.5)(x)
        
        # Flatten and fully connected layer to produce final features.
        x = kl.Flatten()(x)
        x = kl.Dense(self.nb_hidden, kernel_initializer=self.init,
                     kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        
        return self._build(inputs, x)

        

class Dilated5Light(DnaModel):
    def __init__(self, nb_hidden=512, *args, **kwargs):
        super(Dilated5Light, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]
        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        
        # Block 1: 64 filters, no pooling, dilation_rate = 2, dropout 0.1
        x = kl.Conv1D(64, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Conv1D(64, 8, dilation_rate=2, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(0.1)(x)
        
        # Block 2: 128 filters, max pool size = 2, dilation_rate = 4, dropout 0.2
        x = kl.Conv1D(128, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(2)(x)
        x = kl.Conv1D(128, 8, dilation_rate=4, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(0.2)(x)
        
        # Block 3: 192 filters, max pool size = 2, dilation_rate = 8, dropout 0.2
        x = kl.Conv1D(192, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(2)(x)
        x = kl.Conv1D(192, 8, dilation_rate=8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(0.2)(x)
        
        # Block 4: 256 filters, max pool size = 3, dilation_rate = 8, dropout 0.3
        x = kl.Conv1D(256, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(3)(x)
        x = kl.Conv1D(256, 8, dilation_rate=8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(0.3)(x)
        
        # Block 5: 320 filters, max pool size = 4, dilation_rate = 8, dropout 0.4
        x = kl.Conv1D(320, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)
        x = kl.Conv1D(320, 8, dilation_rate=8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(0.4)(x)
        
        # Flatten and fully connected layer to produce final features.
        x = kl.Flatten()(x)
        x = kl.Dense(self.nb_hidden, kernel_initializer=self.init,
                     kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        
        return self._build(inputs, x)
        
        
class DL5LtNoDO(DnaModel):
    def __init__(self, nb_hidden=512, *args, **kwargs):
        super(DL5LtNoDO, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]
        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        
        # Block 1: 64 filters, no pooling, dilation_rate = 2, dropout 0.1
        x = kl.Conv1D(64, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Conv1D(64, 8, dilation_rate=2, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        # x = kl.Dropout(0.1)(x)
        
        # Block 2: 128 filters, max pool size = 2, dilation_rate = 4, dropout 0.2
        x = kl.Conv1D(128, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(2)(x)
        x = kl.Conv1D(128, 8, dilation_rate=4, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        # x = kl.Dropout(0.2)(x)
        
        # Block 3: 192 filters, max pool size = 2, dilation_rate = 8, dropout 0.2
        x = kl.Conv1D(192, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(2)(x)
        x = kl.Conv1D(192, 8, dilation_rate=8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        # x = kl.Dropout(0.2)(x)
        
        # Block 4: 256 filters, max pool size = 3, dilation_rate = 8, dropout 0.3
        x = kl.Conv1D(256, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)
        x = kl.Conv1D(256, 8, dilation_rate=8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        # x = kl.Dropout(0.3)(x)
        
        # Block 5: 320 filters, max pool size = 4, dilation_rate = 8, dropout 0.4
        x = kl.Conv1D(320, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)
        x = kl.Conv1D(320, 8, dilation_rate=8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        # x = kl.Dropout(0.4)(x)
        
        # Flatten and fully connected layer to produce final features.
        x = kl.Flatten()(x)
        x = kl.Dense(self.nb_hidden, kernel_initializer=self.init,
                     kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        
        return self._build(inputs, x)
  
   
   
class DL5Medium(DnaModel):
    def __init__(self, nb_hidden=512, *args, **kwargs):
        super(DL5Medium, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]
        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        
        # Block 1: 256 filters, no pooling, dilation_rate = 2, dropout 0.1
        x = kl.Conv1D(256, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Conv1D(256, 8, dilation_rate=2, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(0.1)(x)
        
        # Block 2: 384 filters, max pool size = 2, dilation_rate = 4, dropout 0.2
        x = kl.Conv1D(384, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(2)(x)
        x = kl.Conv1D(384, 8, dilation_rate=4, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(0.2)(x)
        
        # Block 3: 512 filters, max pool size = 2, dilation_rate = 8, dropout 0.2
        x = kl.Conv1D(512, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(2)(x)
        x = kl.Conv1D(512, 8, dilation_rate=8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(0.2)(x)
        
        # Block 4: 640 filters, max pool size = 3, dilation_rate = 8, dropout 0.3
        x = kl.Conv1D(640, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(3)(x)
        x = kl.Conv1D(640, 8, dilation_rate=8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(0.3)(x)
        
        # Block 5: 768 filters, max pool size = 4, dilation_rate = 8, dropout 0.4
        x = kl.Conv1D(768, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)
        x = kl.Conv1D(768, 8, dilation_rate=8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(0.4)(x)
        
        # Flatten and fully connected layer to produce final features.
        x = kl.Flatten()(x)
        x = kl.Dense(self.nb_hidden, kernel_initializer=self.init,
                     kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        
        return self._build(inputs, x)




class DL5MedMP1(DnaModel):
    def __init__(self, nb_hidden=512, *args, **kwargs):
        super(DL5MedMP1, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]
        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        
        # Block 1: 256 filters, dilation_rate = 2, dropout 0.1
        x = kl.Conv1D(256, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(2)(x)
        x = kl.Conv1D(256, 8, dilation_rate=2, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(0.1)(x)
        
        # Block 2: 384 filters, max pool size = 2, dilation_rate = 4, dropout 0.2
        x = kl.Conv1D(384, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(2)(x)
        x = kl.Conv1D(384, 8, dilation_rate=4, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(0.2)(x)
        
        # Block 3: 512 filters, max pool size = 2, dilation_rate = 8, dropout 0.2
        x = kl.Conv1D(512, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(2)(x)
        x = kl.Conv1D(512, 8, dilation_rate=8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(0.2)(x)
        
        # Block 4: 640 filters, max pool size = 3, dilation_rate = 8, dropout 0.3
        x = kl.Conv1D(640, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(3)(x)
        x = kl.Conv1D(640, 8, dilation_rate=8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(0.3)(x)
        
        # Block 5: 768 filters, max pool size = 4, dilation_rate = 8, dropout 0.4
        x = kl.Conv1D(768, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)
        x = kl.Conv1D(768, 8, dilation_rate=8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(0.4)(x)
        
        # Flatten and fully connected layer to produce final features.
        x = kl.Flatten()(x)
        x = kl.Dense(self.nb_hidden, kernel_initializer=self.init,
                     kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        
        return self._build(inputs, x)
        
        
        
        
        
class DeepSEA2(DnaModel):
    def __init__(self, nb_hidden=512, *args, **kwargs):
        super(DeepSEA2, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]

        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(256, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)
        x = kl.Dropout(0.2)(x)

        x = kl.Conv1D(384, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)
        x = kl.Dropout(0.2)(x)

        x = kl.Conv1D(512, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        # x = kl.MaxPooling1D(3)(x)
        x = kl.Dropout(0.5)(x)

        x = kl.Flatten()(x)

        x = kl.Dense(self.nb_hidden, kernel_initializer=self.init,
                     kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)

        return self._build(inputs, x)
        
        
        
class DeepSEA3(DnaModel):
    def __init__(self, nb_hidden=512, *args, **kwargs):
        super(DeepSEA3, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]

        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(256, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)
        x = kl.Dropout(0.2)(x)

        x = kl.Conv1D(384, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)
        x = kl.Dropout(0.2)(x)

        x = kl.Conv1D(512, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        # x = kl.MaxPooling1D(3)(x)
        x = kl.Dropout(0.2)(x)

        x = kl.Flatten()(x)

        x = kl.Dense(self.nb_hidden, kernel_initializer=self.init,
                     kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)

        return self._build(inputs, x)
        

class DeepSEA4(DnaModel):
    def __init__(self, nb_hidden=512, *args, **kwargs):
        super(DeepSEA4, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]

        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(256, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)
        # x = kl.Dropout(0.2)(x)

        x = kl.Conv1D(384, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)
        # x = kl.Dropout(0.2)(x)

        x = kl.Conv1D(512, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        # x = kl.MaxPooling1D(3)(x)
        # x = kl.Dropout(0.2)(x)

        x = kl.Flatten()(x)

        x = kl.Dense(self.nb_hidden, kernel_initializer=self.init,
                     kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)

        return self._build(inputs, x)
        
        
        
class DeepSEA5(DnaModel):
    def __init__(self, nb_hidden=512, *args, **kwargs):
        super(DeepSEA5, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]

        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(256, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)
        x = kl.Dropout(0.1)(x)

        x = kl.Conv1D(384, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)
        x = kl.Dropout(0.1)(x)

        x = kl.Conv1D(512, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        # x = kl.MaxPooling1D(3)(x)
        x = kl.Dropout(0.1)(x)

        x = kl.Flatten()(x)

        x = kl.Dense(self.nb_hidden, kernel_initializer=self.init,
                     kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)

        return self._build(inputs, x)
        
        
class DeepSEA3Hyb(DnaModel):
    def __init__(self, nb_hidden=512, num_heads=16, ff_dim=512, num_transformer_blocks=2, *args, **kwargs):
        super(DeepSEA3Hyb, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks

    def __call__(self, inputs):
        x = inputs[0]

        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(256, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)
        x = kl.Dropout(0.2)(x)

        x = kl.Conv1D(384, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)
        x = kl.Dropout(0.2)(x)

        x = kl.Conv1D(512, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        # x = kl.MaxPooling1D(3)(x)
        x = kl.Dropout(0.2)(x)   

        
        for _ in range(self.num_transformer_blocks):
            input_x = x
            
            # Multi-head self-attention
            attn_output = kl.MultiHeadAttention(num_heads=self.num_heads, key_dim=32)(x, x)
            attn_output = kl.Dropout(0.3)(attn_output)

            # Add residual connection and normalize
            x = kl.LayerNormalization(epsilon=1e-6)(attn_output + input_x)

            # Feedforward layer
            ff_output = kl.Dense(self.ff_dim, activation="gelu")(x)
            ff_output = kl.Dropout(0.3)(ff_output)

            # Add residual connection and normalize
            x = kl.LayerNormalization(epsilon=1e-6)(ff_output + input_x)
            
        x = kl.Dropout(0.2)(x)

        x = kl.Flatten()(x)

        x = kl.Dense(self.nb_hidden, kernel_initializer=self.init,
                     kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)

        return self._build(inputs, x)
        
        
        


        
        



class RelativePositionMultiHeadAttention(kl.Layer):
    def __init__(self, num_heads, key_dim, max_relative_position=128, **kwargs):
        super(RelativePositionMultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.embed_dim = num_heads * key_dim  # Total embedding dimension after splitting across heads
        self.max_relative_position = max_relative_position
        self.attention = kl.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)

        # Embedding layer for relative positions
        # Maps each relative position to a unique embedding based on the key dimension.
        self.relative_position_embedding_layer = kl.Embedding(
            input_dim=2 * max_relative_position - 1,
            output_dim=key_dim,
            embeddings_initializer="random_normal",
            dtype=tf.float16  # Matching dtype with other inputs and layers
        )

    def build(self, input_shape):
        # Extract dynamic sequence length and embedding dimensions from `input_shape`
        _, seq_length, embed_dim = input_shape

        # Check if the embedding dimension matches expected dimensions.
        if embed_dim is None or embed_dim != self.num_heads * self.key_dim:
            raise ValueError("Embedding dimension must be `num_heads * key_dim`")

        # Calculate and store relative indices for all positions in the sequence
        range_vec = tf.range(seq_length)
        self.relative_indices = tf.expand_dims(range_vec, 1) - tf.expand_dims(range_vec, 0)
        # Clip indices to the range [-max_relative_position, max_relative_position]
        self.relative_indices = tf.clip_by_value(
            self.relative_indices,
            -self.max_relative_position + 1,
            self.max_relative_position - 1
        )
        # Shift indices to start from 0 for compatibility with embedding lookup.
        self.relative_indices += self.max_relative_position - 1

    def call(self, query, value):
        # Cast inputs to `float16` to match the embedding layer's dtype
        query = tf.cast(query, dtype=tf.float16)
        value = tf.cast(value, dtype=tf.float16)

        # Obtain batch size and dynamic sequence length from query shape
        batch_size = tf.shape(query)[0]
        seq_length = tf.shape(query)[1]

        # Reshape `query` and `value` to add `num_heads` dimension for multi-head attention
        query = tf.reshape(query, (batch_size, seq_length, self.num_heads, self.key_dim))
        value = tf.reshape(value, (batch_size, seq_length, self.num_heads, self.key_dim))

        # Gather relative position embeddings for each index pair in the sequence.
        # After expansion, shape becomes (seq_length, seq_length, key_dim).
        rel_pos_emb = self.relative_position_embedding_layer(self.relative_indices)

        # Expand for batch and head dimensions, resulting in (1, seq_length, seq_length, num_heads, key_dim)
        rel_pos_emb = tf.expand_dims(rel_pos_emb, axis=0)
        rel_pos_emb = tf.tile(rel_pos_emb, [batch_size, 1, 1, 1])

        # Add relative position embeddings to the `query`
        query += tf.reduce_sum(rel_pos_emb, axis=2)

        # Reshape `query` and `value` back to (batch_size, seq_length, embed_dim) for multi-head attention.
        query = tf.reshape(query, (batch_size, seq_length, self.embed_dim))
        value = tf.reshape(value, (batch_size, seq_length, self.embed_dim))

        # Apply multi-head attention with the modified `query` and `value`
        attn_output = self.attention(query=query, value=value)
        return attn_output

    def compute_output_shape(self, input_shape):
        # Output shape matches input except for `embed_dim`
        return input_shape[0], input_shape[1], self.embed_dim












class DeepSEA3HybRP(DnaModel):
    def __init__(self, nb_hidden=512, num_heads=16, ff_dim=512, num_transformer_blocks=2, *args, **kwargs):
        super(DeepSEA3HybRP, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks

    def __call__(self, inputs):
        x = inputs[0]
        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)

        # Convolutional layers
        x = kl.Conv1D(256, 8, kernel_initializer=self.init, kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)
        x = kl.Dropout(0.2)(x)

        x = kl.Conv1D(384, 8, kernel_initializer=self.init, kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)
        x = kl.Dropout(0.2)(x)

        x = kl.Conv1D(512, 8, kernel_initializer=self.init, kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(0.2)(x)

        # Transformer blocks with Relative Position MultiHead Attention
        for _ in range(self.num_transformer_blocks):
            input_x = x

            # Multi-head self-attention with relative positions
            attn_output = RelativePositionMultiHeadAttention(num_heads=self.num_heads, key_dim=self.ff_dim // self.num_heads)(x, x)
            attn_output = kl.Dropout(0.3)(attn_output)

            # Add residual connection and normalize
            x = kl.LayerNormalization(epsilon=1e-6)(attn_output + input_x)

            # Feedforward layer
            ff_output = kl.Dense(self.ff_dim, activation="gelu")(x)
            ff_output = kl.Dropout(0.3)(ff_output)

            # Add residual connection and normalize
            x = kl.LayerNormalization(epsilon=1e-6)(ff_output + x)

        # Final dense and activation layers
        x = kl.Dropout(0.2)(x)
        x = kl.Flatten()(x)
        x = kl.Dense(self.nb_hidden, kernel_initializer=self.init, kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)

        return self._build(inputs, x)










  
        






def print_gpu_memory_usage():
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        # Adjust the device name format to remove "/physical_device:" if present
        device_name = gpu.name.split("/physical_device:")[-1]
        memory_info = tf.config.experimental.get_memory_info(device_name)
        print(f"Current Memory Usage on {device_name}: {memory_info['current'] / (1024 ** 2):.2f} MB")
        print(f"Peak Memory Usage on {device_name}: {memory_info['peak'] / (1024 ** 2):.2f} MB")



class AttentionPoolingWithStride(kl.Layer):
    def __init__(self, units, stride=4):
        super(AttentionPoolingWithStride, self).__init__()
        self.units = units
        self.stride = stride
        self.query_dense = kl.Dense(32)
        self.key_dense = kl.Dense(32)
        self.value_dense = kl.Dense(units)
        self.softmax = kl.Softmax(axis=-1)

    def build(self, input_shape):
        super(AttentionPoolingWithStride, self).build(input_shape)

    # @tf.function  # Enables JIT compilation
    def call(self, inputs):
        # print_gpu_memory_usage()  # Print memory usage before operation
        input_shape = tf.shape(inputs)
        batch_size, seq_length, feature_dim = input_shape[0], input_shape[1], input_shape[2]

        # Compute padding for the sequence length to be divisible by stride
        padding_needed = (self.stride - tf.math.mod(seq_length, self.stride)) % self.stride
        inputs = tf.pad(inputs, [[0, 0], [0, padding_needed], [0, 0]])

        # Calculate new sequence length
        new_seq_length = (seq_length + padding_needed) // self.stride

        # Reshape for pooling
        inputs = tf.reshape(inputs, [batch_size, new_seq_length, self.stride, feature_dim])

        # Apply attention mechanism
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        scores = tf.einsum("bnsd,bnsd->bns", query, key)
        attention_weights = self.softmax(scores)
        context = tf.einsum("bns,bnsd->bnd", attention_weights, value)
        # print_gpu_memory_usage()  # Print memory usage after operation
        return context

    def compute_output_shape(self, input_shape):
        # Compute output shape based on the stride and units
        batch_size, seq_length, feature_dim = input_shape
        new_seq_length = (seq_length + (self.stride - (seq_length % self.stride)) % self.stride) // self.stride
        return (batch_size, new_seq_length, self.units)


class DeepSEA3HybAtnPl(DnaModel):
    def __init__(self, nb_hidden=512, num_heads=16, ff_dim=512, num_transformer_blocks=2, *args, **kwargs):
        super(DeepSEA3HybAtnPl, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks

    def __call__(self, inputs):
        x = inputs[0]
        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        
        # Layer 1 with AttentionPooling
        x = kl.Conv1D(256, 8, kernel_initializer=self.init, kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = AttentionPoolingWithStride(units=256, stride=4)(x)  
        x = kl.Dropout(0.2)(x)

        # Layer 2 with AttentionPooling
        x = kl.Conv1D(384, 8, kernel_initializer=self.init, kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = AttentionPoolingWithStride(units=384, stride=4)(x)  
        x = kl.Dropout(0.2)(x)

        # Layer 3
        x = kl.Conv1D(512, 8, kernel_initializer=self.init, kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(0.2)(x)

        # Transformer blocks
        for _ in range(self.num_transformer_blocks):
            input_x = x
            attn_output = kl.MultiHeadAttention(num_heads=self.num_heads, key_dim=32)(x, x)
            attn_output = kl.Dropout(0.3)(attn_output)

            x = kl.LayerNormalization(epsilon=1e-6)(attn_output + input_x)
            ff_output = kl.Dense(self.ff_dim, activation="gelu")(x)
            ff_output = kl.Dropout(0.3)(ff_output)
            x = kl.LayerNormalization(epsilon=1e-6)(ff_output + input_x)
            
        # Flatten and Dense layers
        x = kl.Dropout(0.2)(x)
        x = kl.Flatten()(x)  # Ensure fully defined dimensions before Dense layer
        x = kl.Dense(self.nb_hidden, kernel_initializer=self.init, kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)

        return self._build(inputs, x)





class DeepSEA3HybAB1(DnaModel):
    def __init__(self, nb_hidden=512, num_heads=16, ff_dim=512, num_transformer_blocks=1, *args, **kwargs):
        super(DeepSEA3HybAB1, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks

    def __call__(self, inputs):
        x = inputs[0]

        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(256, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)
        x = kl.Dropout(0.2)(x)

        x = kl.Conv1D(384, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)
        x = kl.Dropout(0.2)(x)

        x = kl.Conv1D(512, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        # x = kl.MaxPooling1D(3)(x)
        x = kl.Dropout(0.2)(x)   

        
        for _ in range(self.num_transformer_blocks):
            input_x = x
            
            # Multi-head self-attention
            attn_output = kl.MultiHeadAttention(num_heads=self.num_heads, key_dim=32)(x, x)
            attn_output = kl.Dropout(0.3)(attn_output)

            # Add residual connection and normalize
            x = kl.LayerNormalization(epsilon=1e-6)(attn_output + input_x)

            # Feedforward layer
            ff_output = kl.Dense(self.ff_dim, activation="gelu")(x)
            ff_output = kl.Dropout(0.3)(ff_output)

            # Add residual connection and normalize
            x = kl.LayerNormalization(epsilon=1e-6)(ff_output + input_x)
            
        x = kl.Dropout(0.2)(x)

        x = kl.Flatten()(x)

        x = kl.Dense(self.nb_hidden, kernel_initializer=self.init,
                     kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)

        return self._build(inputs, x)
        

class DeepSEA3HybAB3(DnaModel):
    def __init__(self, nb_hidden=512, num_heads=16, ff_dim=512, num_transformer_blocks=3, *args, **kwargs):
        super(DeepSEA3HybAB3, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks

    def __call__(self, inputs):
        x = inputs[0]

        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(256, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)
        x = kl.Dropout(0.2)(x)

        x = kl.Conv1D(384, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)
        x = kl.Dropout(0.2)(x)

        x = kl.Conv1D(512, 8, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        # x = kl.MaxPooling1D(3)(x)
        x = kl.Dropout(0.2)(x)   

        
        for _ in range(self.num_transformer_blocks):
            input_x = x
            
            # Multi-head self-attention
            attn_output = kl.MultiHeadAttention(num_heads=self.num_heads, key_dim=32)(x, x)
            attn_output = kl.Dropout(0.3)(attn_output)

            # Add residual connection and normalize
            x = kl.LayerNormalization(epsilon=1e-6)(attn_output + input_x)

            # Feedforward layer
            ff_output = kl.Dense(self.ff_dim, activation="gelu")(x)
            ff_output = kl.Dropout(0.3)(ff_output)

            # Add residual connection and normalize
            x = kl.LayerNormalization(epsilon=1e-6)(ff_output + input_x)
            
        x = kl.Dropout(0.2)(x)

        x = kl.Flatten()(x)

        x = kl.Dense(self.nb_hidden, kernel_initializer=self.init,
                     kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)

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
        

class DanQ(DnaModel):
    def __init__(self, nb_hidden=925, *args, **kwargs):
        super(DanQ, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden
    def __call__(self, inputs):
        x = inputs[0]
        # Define kernel regularizer
        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        # Conv1D + ReLU + MaxPooling + Dropout
        x = kl.Conv1D(320, 26, kernel_initializer=self.init, kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(pool_size=13, strides=13)(x)
        x = kl.Dropout(0.2)(x)
        # Bidirectional LSTM
        x = kl.Bidirectional(kl.LSTM(320, return_sequences=True))(x)
        # Flatten and Dense layers
        x = kl.Flatten()(x)
        x = kl.Dropout(0.5)(x)
        x = kl.Dense(self.nb_hidden, kernel_initializer=self.init, kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        return self._build(inputs, x)
        
class DanQDeep(DnaModel):
    def __init__(self, nb_hidden=925, *args, **kwargs):
        super(DanQDeep, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]
        
        input_length =x.shape[1]  # Assuming input shape is (batch_size, sequence_length, channels)
        
        if input_length >= 32000:
            self.num_blocks = 6
        elif input_length >= 16000:
            self.num_blocks = 5
        elif input_length >= 8000:
            self.num_blocks = 4
        elif input_length >= 4000:
            self.num_blocks = 3
        
        # Define kernel regularizer
        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        filters = [240, 360, 480, 600, 720, 840]
        kernels = [27, 18, 12, 8, 8, 5]
        pools = [4, 4, 3, 3, 2, 2]
        drops = [0.2, 0.2, 0.3, 0.3, 0.4, 0.4]

        # CNN Blocks
        for i in range(self.num_blocks):
            x = kl.Conv1D(filters=filters[i], kernel_size=kernels[i], kernel_initializer=self.init, 
                          kernel_regularizer=kernel_regularizer)(x)
            x = kl.Activation('relu')(x)
            x = kl.MaxPooling1D(pool_size=pools[i], strides=pools[i])(x)
            x = kl.Dropout(drops[i])(x)
        
        # LSTM Blocks
        for i in range(self.num_blocks):
            x = kl.Bidirectional(kl.LSTM(filters[i], return_sequences=True))(x)

        # Flatten and Dense layers
        x = kl.Flatten()(x)
        x = kl.Dropout(0.4)(x)
        
        # Fully connected layer 1
        x = kl.Dense(2048, kernel_initializer=self.init, kernel_regularizer=kernel_regularizer)(x)
        x = kl.ReLU()(x)
        x = kl.Dropout(0.4)(x)

        # Fully connected layer 2
        x = kl.Dense(self.nb_hidden, kernel_initializer=self.init, kernel_regularizer=kernel_regularizer)(x)
        x = kl.ReLU()(x)

        return self._build(inputs, x)
        
        
class DanQDeepRes(DnaModel):
    def __init__(self, nb_hidden=925, *args, **kwargs):
        super(DanQDeepRes, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden
        self.init = "he_normal"  # 'glorot_uniform' or "he_normal"
        self.act = 'relu' # 'relu' or 'elu'

    def __call__(self, inputs):
        x = inputs[0]
        input_length = x.shape[1]  # Assuming input shape is (batch_size, sequence_length, channels)

        if input_length >= 32000:
            self.num_blocks = 6
        elif input_length >= 16000:
            self.num_blocks = 5
        elif input_length >= 8000:
            self.num_blocks = 4
        elif input_length >= 4000:
            self.num_blocks = 3

        # Define kernel regularizer
        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)

        # Adjusted filter sizes, pooling sizes, and dropout values
        filters = [240, 360, 480, 600, 720, 840]
        kernels = [16, 12, 8, 8, 5, 5]
        pools = [4, 4, 4, 4, 2, 2]
        drops = [0.3, 0.3, 0.4, 0.4, 0.5, 0.5]

        # CNN Blocks with skip connections
        for i in range(self.num_blocks):
            shortcut = x  # Save the input for the residual connection
            
            # First convolutional layer with activation
            x = kl.Conv1D(filters=filters[i], kernel_size=kernels[i], padding='same', kernel_initializer=self.init, 
                           kernel_regularizer=kernel_regularizer, activation=self.act)(x)

            # Second convolutional layer with activation
            x = kl.Conv1D(filters=filters[i], kernel_size=kernels[i], padding='same', kernel_initializer=self.init, 
                           kernel_regularizer=kernel_regularizer, activation=self.act)(x)
            
            x = kl.MaxPooling1D(pool_size=pools[i], strides=pools[i])(x)
            x = kl.Dropout(drops[i])(x)

            # Adjust shortcut dimension using a 1x1 convolution
            shortcut = kl.Conv1D(filters=filters[i], kernel_size=1, padding='same', kernel_initializer=self.init,
                                 kernel_regularizer=kernel_regularizer)(shortcut)
            shortcut = kl.MaxPooling1D(pool_size=pools[i], strides=pools[i])(shortcut)
            
            x = kl.add([x, shortcut])  # Residual connection

        # LSTM Blocks with skip connections
        for _ in range(self.num_blocks):
            shortcut = x  # Save the input for the residual connection
            
            x = kl.Bidirectional(kl.LSTM(filters[i] // 2, return_sequences=True))(x)

            # Add a second LSTM layer
            x = kl.Bidirectional(kl.LSTM(filters[i] // 2, return_sequences=True))(x)

            # After every 2 LSTM layers
            x = kl.add([x, shortcut])  # Residual connection

        # Flatten and Dense layers
        x = kl.Flatten()(x)
        x = kl.Dropout(0.4)(x)
        
        # Fully connected layer 1 with activation
        x = kl.Dense(4096, kernel_initializer=self.init, kernel_regularizer=kernel_regularizer, activation=self.act)(x)
        x = kl.Dropout(0.4)(x)

        # Fully connected layer 2 with activation
        x = kl.Dense(self.nb_hidden, kernel_initializer=self.init, kernel_regularizer=kernel_regularizer, activation=self.act)(x)

        return self._build(inputs, x)


class DanQAlt(DnaModel):
    def __init__(self, nb_hidden=925, num_blocks=3, *args, **kwargs):
        super(DanQAlt, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden
        self.num_blocks = num_blocks  # Number of alternating blocks

    def __call__(self, inputs):
        x = inputs[0]
        
        input_length =x.shape[1]  # Assuming input shape is (batch_size, sequence_length, channels)
        
        if input_length >= 32000:
            self.num_blocks = 6
        elif input_length >= 16000:
            self.num_blocks = 5
        elif input_length >= 8000:
            self.num_blocks = 4
        elif input_length >= 4000:
            self.num_blocks = 3
        
        # Define kernel regularizer
        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        filters = [240, 360, 480, 600, 720, 840]
        kernels = [27, 18, 12, 8, 8, 5]
        pools = [4, 4, 3, 3, 2, 2]
        drops = [0.2, 0.2, 0.3, 0.3, 0.4, 0.4]

        for i in range(self.num_blocks):
            # CNN Block
            x = kl.Conv1D(filters=filters[i], kernel_size=kernels[i], kernel_initializer=self.init, 
                          kernel_regularizer=kernel_regularizer)(x)
            x = kl.Activation('relu')(x)
            x = kl.MaxPooling1D(pool_size=pools[i], strides=pools[i])(x)
            x = kl.Dropout(drops[i])(x)

            # Bidirectional LSTM Block
            x = kl.Bidirectional(kl.LSTM(filters[i], return_sequences=True))(x)

        # Flatten and Dense layers
        x = kl.Flatten()(x)
        x = kl.Dropout(0.4)(x)
        
        # Fully connected layer 1
        x = kl.Dense(2048, kernel_initializer=self.init, kernel_regularizer=kernel_regularizer)(x)
        x = kl.ReLU()(x)
        x = kl.Dropout(0.4)(x)

        # Fully connected layer 2
        x = kl.Dense(self.nb_hidden, kernel_initializer=self.init, kernel_regularizer=kernel_regularizer)(x)
        x = kl.ReLU()(x)
        
        return self._build(inputs, x)



class HeartENN(DnaModel):
    def __init__(self, nb_hidden=512, *args, **kwargs):
        super(HeartENN, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden
    
    def __call__(self, inputs):
        x = inputs[0]
        
        # Kernel Regularizer
        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(60, 8, kernel_initializer=self.init, kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Conv1D(60, 8, kernel_initializer=self.init, kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(pool_size=4, strides=4)(x)
        x = kl.BatchNormalization()(x)

        x = kl.Conv1D(80, 8,kernel_initializer=self.init, kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Conv1D(80, 8, kernel_initializer=self.init, kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(pool_size=4, strides=4)(x)
        x = kl.BatchNormalization()(x)
        x = kl.Dropout(0.4)(x)

        x = kl.Conv1D(240, 8, kernel_initializer=self.init, kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.Conv1D(240, 8, kernel_initializer=self.init, kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.BatchNormalization()(x)
        x = kl.Dropout(0.6)(x)

        x = kl.Flatten()(x)

        # Apply Fully connected layers with activation
        x = kl.Dense(self.nb_hidden, kernel_initializer=self.init, kernel_regularizer=kernel_regularizer)(x)
        x = kl.Activation('relu')(x)
        x = kl.BatchNormalization()(x)

        return self._build(inputs, x)
        

class HeartENNOrig(DnaModel):
    def __init__(self, nb_hidden=512, *args, **kwargs):
        """
        Parameters
        ----------
        sequence_length : int
            Length of sequence context on which to train.
        n_genomic_features : int
            The number of chromatin features to predict.
        """
        super(HeartENNOrig, self).__init__(*args, **kwargs)

        conv_kernel_size = 8
        pool_kernel_size = 4

        # Define the ConvNet with L2 regularization
        self.conv_net = tf.keras.Sequential([
            kl.Input(shape=(self.dna_wlen, 4)),
            kl.Conv1D(60, kernel_size=conv_kernel_size, kernel_regularizer=kr.l2(self.l2_decay)),
            kl.ReLU(),
            kl.Conv1D(60, kernel_size=conv_kernel_size, kernel_regularizer=kr.l2(self.l2_decay)),
            kl.ReLU(),
            kl.MaxPooling1D(pool_size=pool_kernel_size, strides=pool_kernel_size),
            kl.BatchNormalization(),

            kl.Conv1D(80, kernel_size=conv_kernel_size, kernel_regularizer=kr.l2(self.l2_decay)),
            kl.ReLU(),
            kl.Conv1D(80, kernel_size=conv_kernel_size, kernel_regularizer=kr.l2(self.l2_decay)),
            kl.ReLU(),
            kl.MaxPooling1D(pool_size=pool_kernel_size, strides=pool_kernel_size),
            kl.BatchNormalization(),
            kl.Dropout(0.4),

            kl.Conv1D(240, kernel_size=conv_kernel_size, kernel_regularizer=kr.l2(self.l2_decay)),
            kl.ReLU(),
            kl.Conv1D(240, kernel_size=conv_kernel_size, kernel_regularizer=kr.l2(self.l2_decay)),
            kl.ReLU(),
            kl.BatchNormalization(),
            kl.Dropout(0.6)
        ])

        # Calculate the output size after the convolutions and pooling
        reduce_by = 2 * (conv_kernel_size - 1)
        pool_kernel_size = float(pool_kernel_size)
        self._n_channels = int(
            np.floor(
                (np.floor(
                    (self.dna_wlen - reduce_by) / pool_kernel_size) - reduce_by) / pool_kernel_size
                ) - reduce_by)

        self.classifier = tf.keras.Sequential([
            #kl.Dense(240 * self._n_channels, activation='relu', kernel_regularizer=kr.l2(self.l2_decay)),
            kl.Dense(4096, activation='relu', kernel_regularizer=kr.l2(self.l2_decay)),
            kl.BatchNormalization(),
            kl.Dense(nb_hidden, activation='relu', kernel_regularizer=kr.l2(self.l2_decay)),
        ])

    def __call__(self, inputs):
        x = inputs[0]
        # Renormalizing weights 
        for layer in self.conv_net.layers:
            if isinstance(layer, tf.keras.layers.Conv1D):
                weights, biases = layer.get_weights()
                # Calculate L2 norm for the weights
                norm = np.linalg.norm(weights.flatten(), ord=2)  # Flatten to avoid dimension issues
                if norm > 0.9:
                    weights *= 0.9 / norm
                layer.set_weights([weights, biases])

        # Renormalizing weights for dense layers
        for layer in self.classifier.layers:
            if isinstance(layer, tf.keras.layers.Dense):
                if layer.weights:  # Check if the layer has weights
                    weights, biases = layer.get_weights()
                    norm = np.linalg.norm(weights.flatten(), ord=2)  # Flatten here too
                    if norm > 0.9:
                        weights *= 0.9 / norm
                    layer.set_weights([weights, biases])
                

        x = self.conv_net(x)
        x = kl.Flatten()(x)
        x = self.classifier(x)
        
        return self._build(inputs, x)

        
class LongerCNN(DnaModel):
    def __init__(self, nb_hidden=512, *args, **kwargs):
        super(LongerCNN, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]
        # Define kernel regularizer
        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)

        # Get the input shape to determine sequence length
        input_length = x.shape[1]  # Get sequence length dynamically

        # Common Convolutional block 1
        x = kl.Conv1D(filters=320, kernel_size=26, kernel_initializer=self.init, 
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.ReLU()(x)
        x = kl.MaxPooling1D(pool_size=9, strides=9)(x)  # Slightly higher than 1/3 of 26
        x = kl.Dropout(0.2)(x)

        # Common Convolutional block 2
        x = kl.Conv1D(filters=480, kernel_size=13, kernel_initializer=self.init, 
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.ReLU()(x)
        x = kl.MaxPooling1D(pool_size=5, strides=5)(x)  # Slightly higher than 1/3 of 13
        x = kl.Dropout(0.3)(x)

        # Common Convolutional block 3
        x = kl.Conv1D(filters=960, kernel_size=7, kernel_initializer=self.init, 
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.ReLU()(x)
        x = kl.MaxPooling1D(pool_size=3, strides=3)(x)  # Slightly higher than 1/3 of 7
        x = kl.Dropout(0.4)(x)

        # Conditional Convolutional block for longer sequences
        if input_length >= 16000:  # Check if input length is at least 16k
            x = kl.Conv1D(filters=1280, kernel_size=10, kernel_initializer=self.init, 
                          kernel_regularizer=kernel_regularizer)(x)
            x = kl.ReLU()(x)
            x = kl.MaxPooling1D(pool_size=3, strides=3)(x)  # Slightly higher than 1/3 of 10
            x = kl.Dropout(0.5)(x)

        # Flatten the 1D convolution output
        x = kl.Flatten()(x)

        # Fully connected layers
        x = kl.Dense(2048 if input_length >= 16000 else 512, kernel_initializer=self.init, 
                     kernel_regularizer=kernel_regularizer)(x)
        x = kl.ReLU()(x)
        x = kl.Dropout(0.5)(x)

        # Second fully connected layer
        x = kl.Dense(self.nb_hidden, kernel_initializer=self.init, kernel_regularizer=kernel_regularizer)(x)
        x = kl.ReLU()(x)
        
        return self._build(inputs, x)


class LongerDNADeeperCNN(DnaModel):
    def __init__(self, nb_hidden=512, *args, **kwargs):
        super(LongerDNADeeperCNN, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]
        # Get the input sequence length run last conv bock optionally for longer sequences
        input_length = x.shape[1]  # Get sequence length dynamically

        # Define kernel regularizer
        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)

        # Convolutional block 1
        x = kl.Conv1D(filters=240, kernel_size=27, kernel_initializer=self.init, 
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.ReLU()(x)
        x = kl.MaxPooling1D(pool_size=4, strides=4)(x)  
        x = kl.Dropout(0.2)(x)

        # Convolutional block 2
        x = kl.Conv1D(filters=360, kernel_size=18, kernel_initializer=self.init, 
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.ReLU()(x)
        x = kl.MaxPooling1D(pool_size=4, strides=4)(x)  
        x = kl.Dropout(0.3)(x)

        # Convolutional block 3
        x = kl.Conv1D(filters=480, kernel_size=12, kernel_initializer=self.init, 
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.ReLU()(x)
        x = kl.MaxPooling1D(pool_size=3, strides=3)(x)  
        x = kl.Dropout(0.3)(x)

        # Convolutional block 4
        x = kl.Conv1D(filters=600, kernel_size=8, kernel_initializer=self.init, 
                      kernel_regularizer=kernel_regularizer)(x)
        x = kl.ReLU()(x)
        
        # Convolutional block 5: Runs for sequence length 8k
        if (input_length > 4096) and (input_length < 16000):
            x = kl.MaxPooling1D(pool_size=2, strides=2)(x) 
            x = kl.Dropout(0.3)(x)
            x = kl.Conv1D(filters=720, kernel_size=8, kernel_initializer=self.init, 
                          kernel_regularizer=kernel_regularizer)(x)
            x = kl.ReLU()(x)

        # Convolutional block 6 and 7: Runs for sequence length 16k
        if (input_length > 8192) and (input_length < 32000):
            x = kl.MaxPooling1D(pool_size=2, strides=2)(x)
            x = kl.Dropout(0.3)(x)
            x = kl.Conv1D(filters=840, kernel_size=5, kernel_initializer=self.init, 
                          kernel_regularizer=kernel_regularizer)(x)
            x = kl.ReLU()(x)
            
            x = kl.MaxPooling1D(pool_size=2, strides=2)(x)
            x = kl.Dropout(0.3)(x)
            x = kl.Conv1D(filters=960, kernel_size=5, kernel_initializer=self.init, 
                          kernel_regularizer=kernel_regularizer)(x)
            x = kl.ReLU()(x)
        
        # Convolutional block 8, 9 and 10: Runs for sequence length at least 32k  
        if input_length >= 32000: 
            x = kl.MaxPooling1D(pool_size=2, strides=2)(x)
            x = kl.Dropout(0.3)(x)
            x = kl.Conv1D(filters=1080, kernel_size=3, kernel_initializer=self.init, 
                          kernel_regularizer=kernel_regularizer)(x)
            x = kl.ReLU()(x)
            
            x = kl.MaxPooling1D(pool_size=2, strides=2)(x)
            x = kl.Dropout(0.3)(x)
            x = kl.Conv1D(filters=1200, kernel_size=3, kernel_initializer=self.init, 
                          kernel_regularizer=kernel_regularizer)(x)
            x = kl.ReLU()(x)
            
            x = kl.MaxPooling1D(pool_size=2, strides=2)(x)
            x = kl.Dropout(0.3)(x)
            x = kl.Conv1D(filters=1320, kernel_size=3, kernel_initializer=self.init, 
                          kernel_regularizer=kernel_regularizer)(x)
            x = kl.ReLU()(x)


        x = kl.Dropout(0.4)(x)
        # Flatten the 1D convolution output
        x = kl.Flatten()(x)

        # Fully connected layer 1
        x = kl.Dense(2048, kernel_initializer=self.init, kernel_regularizer=kernel_regularizer)(x)
        x = kl.ReLU()(x)
        x = kl.Dropout(0.4)(x)

        # Fully connected layer 2
        x = kl.Dense(self.nb_hidden, kernel_initializer=self.init, kernel_regularizer=kernel_regularizer)(x)
        x = kl.ReLU()(x)
        x = kl.Dropout(0.4)(x)
        
        return self._build(inputs, x)


  
        
        
class TranCNN(DnaModel):
    def __init__(self, nb_hidden=512, num_heads=16, ff_dim=512, add_conv_blocks=1, num_transformer_blocks=2, *args, **kwargs):
        super(TranCNN, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.add_conv_blocks = add_conv_blocks
        self.init = "he_normal"
        self.act = gelu

    def __call__(self, inputs):
        x = inputs[0]
        input_length = x.shape[1]

        # Determine num_transformer_blocks based on input_length
        if input_length >= 32000:
            self.add_conv_blocks = 4
            self.num_transformer_blocks = 9
        elif input_length >= 16000:
            self.add_conv_blocks = 3
            self.num_transformer_blocks = 8
        elif input_length >= 8000:
            self.add_conv_blocks = 2
            self.num_transformer_blocks = 7
        elif input_length >= 4000:
            self.add_conv_blocks = 1
            self.num_transformer_blocks = 6

        # Define kernel regularizer
        kernel_regularizer = kr.L1L2(l1=0.001, l2=0.001)  # Adjust l1_decay and l2_decay as needed

        # Stem
        x = kl.Conv1D(filters=256, kernel_size=15, kernel_initializer=self.init,
                       kernel_regularizer=kernel_regularizer, padding='same', activation=self.act)(x)

        # Convolutional block 1
        shortcut = x
        x = kl.Conv1D(filters=256, kernel_size=5, kernel_initializer=self.init,
                       kernel_regularizer=kernel_regularizer, padding='same', activation=self.act)(x)
        x = kl.BatchNormalization()(x)
        x = kl.MaxPooling1D(pool_size=2, strides=2)(x)
        x = kl.Dropout(0.2)(x)
        shortcut = kl.Conv1D(filters=256, kernel_size=1, kernel_initializer=self.init, padding='same')(shortcut) # Adjust shortcut dimension
        shortcut = kl.MaxPooling1D(pool_size=2, strides=2)(shortcut)  # Move pooling after Conv
        x = kl.add([x, shortcut])

        # Convolutional block 2
        shortcut = x
        x = kl.Conv1D(filters=384, kernel_size=5, kernel_initializer=self.init,
                       kernel_regularizer=kernel_regularizer, padding='same', activation=self.act)(x)
        x = kl.BatchNormalization()(x)
        x = kl.MaxPooling1D(pool_size=2, strides=2)(x)
        x = kl.Dropout(0.2)(x)
        shortcut = kl.Conv1D(filters=384, kernel_size=1, kernel_initializer=self.init, padding='same')(shortcut) # Adjust shortcut dimension
        shortcut = kl.MaxPooling1D(pool_size=2, strides=2)(shortcut)  # Move pooling after Conv
        x = kl.add([x, shortcut])
        
        # Convolutional block 3
        shortcut = x
        x = kl.Conv1D(filters=384, kernel_size=5, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer, padding='same', activation=self.act)(x)
        x = kl.BatchNormalization()(x)
        x = kl.MaxPooling1D(pool_size=2, strides=2)(x)
        x = kl.Dropout(0.2)(x)
        shortcut = kl.Conv1D(filters=384, kernel_size=1, kernel_initializer=self.init, padding='same')(shortcut) # Adjust shortcut dimension
        shortcut = kl.MaxPooling1D(pool_size=2, strides=2)(shortcut)  # Move pooling after Conv
        x = kl.add([x, shortcut])
        
        # Convolutional block 4
        shortcut = x
        x = kl.Conv1D(filters=384, kernel_size=5, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer, padding='same', activation=self.act)(x)
        x = kl.BatchNormalization()(x)
        x = kl.MaxPooling1D(pool_size=2, strides=2)(x)
        x = kl.Dropout(0.2)(x)
        shortcut = kl.Conv1D(filters=384, kernel_size=1, kernel_initializer=self.init, padding='same')(shortcut) # Adjust shortcut dimension
        shortcut = kl.MaxPooling1D(pool_size=2, strides=2)(shortcut)  # Move pooling after Conv
        x = kl.add([x, shortcut])

        # Additional conv blocks 
        if self.add_conv_blocks:
            for _ in range(self.add_conv_blocks):
                shortcut = x
                x = kl.Conv1D(filters=512, kernel_size=5, kernel_initializer=self.init,
                               kernel_regularizer=kernel_regularizer, padding='same', activation=self.act)(x)
                x = kl.BatchNormalization()(x)
                x = kl.MaxPooling1D(pool_size=2, strides=2)(x)
                x = kl.Dropout(0.3)(x)
                shortcut = kl.Conv1D(filters=512, kernel_size=1, kernel_initializer=self.init, padding='same')(shortcut) # Adjust shortcut dimension
                shortcut = kl.MaxPooling1D(pool_size=2, strides=2)(shortcut)  # Move pooling after Conv
                x = kl.add([x, shortcut])


        # Reshape to pass into Transformer blocks
        seq_len = x.shape[1]
        x = kl.Reshape((seq_len, 512))(x)

        for _ in range(self.num_transformer_blocks):
            input_x = x
            
            # Multi-head self-attention
            attn_output = kl.MultiHeadAttention(num_heads=self.num_heads, key_dim=32)(x, x)
            attn_output = kl.Dropout(0.3)(attn_output)

            # Add residual connection and normalize
            x = kl.LayerNormalization(epsilon=1e-6)(attn_output + input_x)

            # Feedforward layer
            ff_output = kl.Dense(self.ff_dim, activation="gelu")(x)
            ff_output = kl.Dropout(0.3)(ff_output)

            # Add residual connection and normalize
            x = kl.LayerNormalization(epsilon=1e-6)(ff_output + input_x)

        # # Reduce dimension by 1x1 convolution
        # x = kl.Conv1D(filters=512, kernel_size=1, kernel_initializer=self.init, padding='same', activation=self.act)(x)
        # # Flatten the transformer output
        x = kl.Flatten()(x)

        # Fully connected layers with combined activation
        x = kl.Dense(4096, kernel_initializer=self.init, kernel_regularizer=kernel_regularizer, activation=self.act)(x)
        x = kl.Dropout(0.4)(x)

        x = kl.Dense(self.nb_hidden, kernel_initializer=self.init, kernel_regularizer=kernel_regularizer, activation=self.act)(x)
        x = kl.Dropout(0.4)(x)

        return self._build(inputs, x)
        
        
        
class TranCNNLean(DnaModel):
    def __init__(self, nb_hidden=512, num_heads=8, ff_dim=256, add_conv_blocks=1, num_transformer_blocks=2, *args, **kwargs):
        super(TranCNNLean, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.add_conv_blocks = add_conv_blocks
        self.init = "he_normal"
        self.act = gelu

    def __call__(self, inputs):
        x = inputs[0]
        input_length = x.shape[1]

        # Determine num_transformer_blocks based on input_length
        if input_length >= 32000:
            self.add_conv_blocks = 4
            self.num_transformer_blocks = 9
        elif input_length >= 16000:
            self.add_conv_blocks = 3
            self.num_transformer_blocks = 8
        elif input_length >= 8000:
            self.add_conv_blocks = 2
            self.num_transformer_blocks = 7
        elif input_length >= 4000:
            self.add_conv_blocks = 1
            self.num_transformer_blocks = 6

        # Define kernel regularizer
        kernel_regularizer = kr.L1L2(l1=0.001, l2=0.001)  # Adjust l1_decay and l2_decay as needed

        # Stem
        x = kl.Conv1D(filters=128, kernel_size=15, kernel_initializer=self.init,
                       kernel_regularizer=kernel_regularizer, padding='same', activation=self.act)(x)

        # Convolutional block 1
        shortcut = x
        x = kl.Conv1D(filters=128, kernel_size=5, kernel_initializer=self.init,
                       kernel_regularizer=kernel_regularizer, padding='same', activation=self.act)(x)
        x = kl.BatchNormalization()(x)
        x = kl.MaxPooling1D(pool_size=2, strides=2)(x)
        x = kl.Dropout(0.2)(x)
        shortcut = kl.Conv1D(filters=128, kernel_size=1, kernel_initializer=self.init, padding='same')(shortcut) # Adjust shortcut dimension
        shortcut = kl.MaxPooling1D(pool_size=2, strides=2)(shortcut)  # Move pooling after Conv
        x = kl.add([x, shortcut])

        # Convolutional block 2
        shortcut = x
        x = kl.Conv1D(filters=256, kernel_size=5, kernel_initializer=self.init,
                       kernel_regularizer=kernel_regularizer, padding='same', activation=self.act)(x)
        x = kl.BatchNormalization()(x)
        x = kl.MaxPooling1D(pool_size=2, strides=2)(x)
        x = kl.Dropout(0.2)(x)
        shortcut = kl.Conv1D(filters=256, kernel_size=1, kernel_initializer=self.init, padding='same')(shortcut) # Adjust shortcut dimension
        shortcut = kl.MaxPooling1D(pool_size=2, strides=2)(shortcut)  # Move pooling after Conv
        x = kl.add([x, shortcut])
        
        # Convolutional block 3
        shortcut = x
        x = kl.Conv1D(filters=256, kernel_size=5, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer, padding='same', activation=self.act)(x)
        x = kl.BatchNormalization()(x)
        x = kl.MaxPooling1D(pool_size=2, strides=2)(x)
        x = kl.Dropout(0.2)(x)
        shortcut = kl.Conv1D(filters=256, kernel_size=1, kernel_initializer=self.init, padding='same')(shortcut) # Adjust shortcut dimension
        shortcut = kl.MaxPooling1D(pool_size=2, strides=2)(shortcut)  # Move pooling after Conv
        x = kl.add([x, shortcut])
        
        # Convolutional block 4
        shortcut = x
        x = kl.Conv1D(filters=384, kernel_size=5, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer, padding='same', activation=self.act)(x)
        x = kl.BatchNormalization()(x)
        x = kl.MaxPooling1D(pool_size=2, strides=2)(x)
        x = kl.Dropout(0.2)(x)
        shortcut = kl.Conv1D(filters=384, kernel_size=1, kernel_initializer=self.init, padding='same')(shortcut) # Adjust shortcut dimension
        shortcut = kl.MaxPooling1D(pool_size=2, strides=2)(shortcut)  # Move pooling after Conv
        x = kl.add([x, shortcut])

        # Additional conv blocks 
        if self.add_conv_blocks:
            for _ in range(self.add_conv_blocks):
                shortcut = x
                x = kl.Conv1D(filters=256, kernel_size=5, kernel_initializer=self.init,
                               kernel_regularizer=kernel_regularizer, padding='same', activation=self.act)(x)
                x = kl.BatchNormalization()(x)
                x = kl.MaxPooling1D(pool_size=2, strides=2)(x)
                x = kl.Dropout(0.3)(x)
                shortcut = kl.Conv1D(filters=256, kernel_size=1, kernel_initializer=self.init, padding='same')(shortcut) # Adjust shortcut dimension
                shortcut = kl.MaxPooling1D(pool_size=2, strides=2)(shortcut)  # Move pooling after Conv
                x = kl.add([x, shortcut])
                
                # shortcut = x
                # x = kl.Conv1D(filters=256, kernel_size=5, kernel_initializer=self.init,
                #               kernel_regularizer=kernel_regularizer, padding='same', activation=self.act)(x)
                # x = kl.BatchNormalization()(x)
                # x = kl.MaxPooling1D(pool_size=2, strides=2)(x)
                # x = kl.Dropout(0.3)(x)
                # shortcut = kl.Conv1D(filters=256, kernel_size=1, kernel_initializer=self.init, padding='same')(shortcut) # Adjust shortcut dimension
                # shortcut = kl.MaxPooling1D(pool_size=2, strides=2)(shortcut)  # Move pooling after Conv
                # x = kl.add([x, shortcut])


        # Reshape to pass into Transformer blocks
        seq_len = x.shape[1]
        x = kl.Reshape((seq_len, 256))(x)

        for _ in range(self.num_transformer_blocks):
            input_x = x
            
            # Multi-head self-attention
            attn_output = kl.MultiHeadAttention(num_heads=self.num_heads, key_dim=32)(x, x)
            attn_output = kl.Dropout(0.3)(attn_output)

            # Add residual connection and normalize
            x = kl.LayerNormalization(epsilon=1e-6)(attn_output + input_x)

            # Feedforward layer
            ff_output = kl.Dense(self.ff_dim, activation="gelu")(x)
            ff_output = kl.Dropout(0.3)(ff_output)

            # Add residual connection and normalize
            x = kl.LayerNormalization(epsilon=1e-6)(ff_output + input_x)

        # # Reduce dimension by 1x1 convolution
        x = kl.Conv1D(filters=128, kernel_size=1, kernel_initializer=self.init, padding='same', activation=self.act)(x)
        # # Flatten the transformer output
        x = kl.Flatten()(x)

        # Fully connected layers with combined activation
        x = kl.Dense(4096, kernel_initializer=self.init, kernel_regularizer=kernel_regularizer, activation=self.act)(x)
        x = kl.Dropout(0.4)(x)

        x = kl.Dense(self.nb_hidden, kernel_initializer=self.init, kernel_regularizer=kernel_regularizer, activation=self.act)(x)
        x = kl.Dropout(0.4)(x)

        return self._build(inputs, x)
        
      
        
class TranCNNLean2(DnaModel):
    def __init__(self, nb_hidden=512, num_heads=8, ff_dim=256, add_conv_blocks=1, num_transformer_blocks=2, *args, **kwargs):
        super(TranCNNLean2, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.add_conv_blocks = add_conv_blocks
        self.init = "he_normal"
        self.act = gelu

    def __call__(self, inputs):
        x = inputs[0]
        input_length = x.shape[1]

        # Determine num_transformer_blocks based on input_length
        if input_length >= 32000:
            self.add_conv_blocks = 4
            self.num_transformer_blocks = 9
        elif input_length >= 16000:
            self.add_conv_blocks = 3
            self.num_transformer_blocks = 8
        elif input_length >= 8000:
            self.add_conv_blocks = 2
            self.num_transformer_blocks = 7
        elif input_length >= 4000:
            self.add_conv_blocks = 1
            self.num_transformer_blocks = 6

        # Define kernel regularizer
        kernel_regularizer = kr.L1L2(l1=0.001, l2=0.001)  # Adjust l1_decay and l2_decay as needed

        # Stem
        x = kl.Conv1D(filters=128, kernel_size=15, kernel_initializer=self.init,
                       kernel_regularizer=kernel_regularizer, padding='same', activation=self.act)(x)

        # Convolutional block 1
        shortcut = x
        x = kl.Conv1D(filters=128, kernel_size=5, kernel_initializer=self.init,
                       kernel_regularizer=kernel_regularizer, padding='same', activation=self.act)(x)
        x = kl.BatchNormalization()(x)
        x = kl.MaxPooling1D(pool_size=2, strides=2)(x)
        x = kl.Dropout(0.2)(x)
        shortcut = kl.Conv1D(filters=128, kernel_size=1, kernel_initializer=self.init, padding='same')(shortcut) # Adjust shortcut dimension
        shortcut = kl.MaxPooling1D(pool_size=2, strides=2)(shortcut)  # Move pooling after Conv
        x = kl.add([x, shortcut])

        # Convolutional block 2
        shortcut = x
        x = kl.Conv1D(filters=256, kernel_size=5, kernel_initializer=self.init,
                       kernel_regularizer=kernel_regularizer, padding='same', activation=self.act)(x)
        x = kl.BatchNormalization()(x)
        x = kl.MaxPooling1D(pool_size=2, strides=2)(x)
        x = kl.Dropout(0.2)(x)
        shortcut = kl.Conv1D(filters=256, kernel_size=1, kernel_initializer=self.init, padding='same')(shortcut) # Adjust shortcut dimension
        shortcut = kl.MaxPooling1D(pool_size=2, strides=2)(shortcut)  # Move pooling after Conv
        x = kl.add([x, shortcut])
        
        # Convolutional block 3
        shortcut = x
        x = kl.Conv1D(filters=256, kernel_size=5, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer, padding='same', activation=self.act)(x)
        x = kl.BatchNormalization()(x)
        x = kl.MaxPooling1D(pool_size=2, strides=2)(x)
        x = kl.Dropout(0.2)(x)
        shortcut = kl.Conv1D(filters=256, kernel_size=1, kernel_initializer=self.init, padding='same')(shortcut) # Adjust shortcut dimension
        shortcut = kl.MaxPooling1D(pool_size=2, strides=2)(shortcut)  # Move pooling after Conv
        x = kl.add([x, shortcut])
        
        # Convolutional block 4
        shortcut = x
        x = kl.Conv1D(filters=256, kernel_size=5, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer, padding='same', activation=self.act)(x)
        x = kl.BatchNormalization()(x)
        x = kl.MaxPooling1D(pool_size=2, strides=2)(x)
        x = kl.Dropout(0.2)(x)
        shortcut = kl.Conv1D(filters=256, kernel_size=1, kernel_initializer=self.init, padding='same')(shortcut) # Adjust shortcut dimension
        shortcut = kl.MaxPooling1D(pool_size=2, strides=2)(shortcut)  # Move pooling after Conv
        x = kl.add([x, shortcut])
        
        # Convolutional block 5
        shortcut = x
        x = kl.Conv1D(filters=256, kernel_size=5, kernel_initializer=self.init,
                      kernel_regularizer=kernel_regularizer, padding='same', activation=self.act)(x)
        x = kl.BatchNormalization()(x)
        x = kl.MaxPooling1D(pool_size=2, strides=2)(x)
        x = kl.Dropout(0.2)(x)
        shortcut = kl.Conv1D(filters=256, kernel_size=1, kernel_initializer=self.init, padding='same')(shortcut) # Adjust shortcut dimension
        shortcut = kl.MaxPooling1D(pool_size=2, strides=2)(shortcut)  # Move pooling after Conv
        x = kl.add([x, shortcut])

        # Additional conv blocks 
        if self.add_conv_blocks:
            for _ in range(self.add_conv_blocks):
                shortcut = x
                x = kl.Conv1D(filters=256, kernel_size=5, kernel_initializer=self.init,
                               kernel_regularizer=kernel_regularizer, padding='same', activation=self.act)(x)
                x = kl.BatchNormalization()(x)
                x = kl.MaxPooling1D(pool_size=2, strides=2)(x)
                x = kl.Dropout(0.3)(x)
                shortcut = kl.Conv1D(filters=256, kernel_size=1, kernel_initializer=self.init, padding='same')(shortcut) # Adjust shortcut dimension
                shortcut = kl.MaxPooling1D(pool_size=2, strides=2)(shortcut)  # Move pooling after Conv
                x = kl.add([x, shortcut])
                
                # shortcut = x
                # x = kl.Conv1D(filters=256, kernel_size=5, kernel_initializer=self.init,
                #               kernel_regularizer=kernel_regularizer, padding='same', activation=self.act)(x)
                # x = kl.BatchNormalization()(x)
                # x = kl.MaxPooling1D(pool_size=2, strides=2)(x)
                # x = kl.Dropout(0.3)(x)
                # shortcut = kl.Conv1D(filters=256, kernel_size=1, kernel_initializer=self.init, padding='same')(shortcut) # Adjust shortcut dimension
                # shortcut = kl.MaxPooling1D(pool_size=2, strides=2)(shortcut)  # Move pooling after Conv
                # x = kl.add([x, shortcut])


        # Reshape to pass into Transformer blocks
        seq_len = x.shape[1]
        x = kl.Reshape((seq_len, 256))(x)

        for _ in range(self.num_transformer_blocks):
            input_x = x
            
            # Multi-head self-attention
            attn_output = kl.MultiHeadAttention(num_heads=self.num_heads, key_dim=32)(x, x)
            attn_output = kl.Dropout(0.3)(attn_output)

            # Add residual connection and normalize
            x = kl.LayerNormalization(epsilon=1e-6)(attn_output + input_x)

            # Feedforward layer
            ff_output = kl.Dense(self.ff_dim, activation="gelu")(x)
            ff_output = kl.Dropout(0.3)(ff_output)

            # Add residual connection and normalize
            x = kl.LayerNormalization(epsilon=1e-6)(ff_output + input_x)

        # # Reduce dimension by 1x1 convolution
        # x = kl.Conv1D(filters=128, kernel_size=1, kernel_initializer=self.init, padding='same', activation=self.act)(x)
        # # Flatten the transformer output
        x = kl.Flatten()(x)

        # Fully connected layers with combined activation
        x = kl.Dense(4096, kernel_initializer=self.init, kernel_regularizer=kernel_regularizer, activation=self.act)(x)
        x = kl.Dropout(0.4)(x)

        x = kl.Dense(self.nb_hidden, kernel_initializer=self.init, kernel_regularizer=kernel_regularizer, activation=self.act)(x)
        x = kl.Dropout(0.4)(x)

        return self._build(inputs, x)
        
    

 
 
class TranCNNAlt(DnaModel):
    def __init__(self, nb_hidden=512, num_heads=16, ff_dim=512, num_transformer_blocks=2, *args, **kwargs):
        super(TranCNNAlt, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.init = "he_normal" # 'glorot_uniform' or "he_normal"
        self.act = kl.ELU() # kl.ReLU() or kl.ELU() or kl.Activation('swish')

    def __call__(self, inputs):
        x = inputs[0]
        
        input_length = x.shape[1]
        
        # Determine num_transformer_blocks
        if input_length >= 32000:
            self.num_transformer_blocks = 5
        elif input_length >= 16000:
            self.num_transformer_blocks = 4
        elif input_length >= 8000:
            self.num_transformer_blocks = 3
        elif input_length >= 4000:
            self.num_transformer_blocks = 2
        # Define kernel regularizer
        
        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)

        # Convolutional block 1
        x = kl.Conv1D(filters=360, kernel_size=27, kernel_initializer=self.init, 
                      kernel_regularizer=kernel_regularizer)(x)
        x = self.act(x)
        x = kl.BatchNormalization()(x) 
        x = kl.MaxPooling1D(pool_size=4, strides=4)(x)  
        x = kl.Dropout(0.3)(x)

        # Convolutional block 2
        x = kl.Conv1D(filters=480, kernel_size=18, kernel_initializer=self.init, 
                      kernel_regularizer=kernel_regularizer)(x)
        x = self.act(x)
        x = kl.BatchNormalization()(x) 
        x = kl.MaxPooling1D(pool_size=4, strides=4)(x)  
        x = kl.Dropout(0.3)(x)

        # Start alternating CNN and Transformer blocks
        for _ in range(self.num_transformer_blocks):
            # CNN block
            x = kl.Conv1D(filters=512, kernel_size=8, kernel_initializer=self.init,
                          kernel_regularizer=kernel_regularizer)(x)
            x = self.act(x)
            x = kl.BatchNormalization()(x) 
            x = kl.MaxPooling1D(pool_size=2, strides=2)(x)
            x = kl.Dropout(0.4)(x)

            # Reshape to pass into Transformer blocks
            seq_len = x.shape[1]  # Updated sequence length after convolutions
            x = kl.Reshape((seq_len, 512))(x)

            # Transformer block
            attn_output = kl.MultiHeadAttention(num_heads=self.num_heads, key_dim=32)(x, x)
            attn_output = kl.Dropout(0.3)(attn_output)
            attn_output = kl.LayerNormalization(epsilon=1e-6)(attn_output + x)
            # Feedforward layer
            ff_output = kl.Dense(self.ff_dim, activation="gelu")(attn_output)
            ff_output = kl.Dropout(0.4)(ff_output)
            ff_output = kl.LayerNormalization(epsilon=1e-6)(ff_output + attn_output)

            x = ff_output
            
        # Additional Transformer Blocks
        # Reshape to pass into Transformer blocks
        seq_len = x.shape[1]  # Updated sequence length after convolutions
        x = kl.Reshape((seq_len, 512))(x)
        attn_output = kl.MultiHeadAttention(num_heads=self.num_heads, key_dim=32)(x, x)
        attn_output = kl.Dropout(0.5)(attn_output)
        attn_output = kl.LayerNormalization(epsilon=1e-6)(attn_output + x)

        # Feedforward layer
        ff_output = kl.Dense(self.ff_dim, activation="gelu")(attn_output)
        ff_output = kl.Dropout(0.5)(ff_output)
        ff_output = kl.LayerNormalization(epsilon=1e-6)(ff_output + attn_output)
        x = ff_output

        # Flatten the transformer output
        x = kl.Flatten()(x)

        # Fully connected layers
        x = kl.Dense(2048, kernel_initializer=self.init, kernel_regularizer=kernel_regularizer)(x)
        x = self.act(x)
        x = kl.Dropout(0.5)(x)

        x = kl.Dense(self.nb_hidden, kernel_initializer=self.init, kernel_regularizer=kernel_regularizer)(x)
        x = self.act(x)
        x = kl.Dropout(0.5)(x)

        return self._build(inputs, x)
        
        
        
class TranCNNAltRes(DnaModel):
    def __init__(self, nb_hidden=512, num_heads=16, ff_dim=512, num_transformer_blocks=2, *args, **kwargs):
        super(TranCNNAltRes, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.init = "he_normal"  # 'glorot_uniform' or "he_normal"
        self.act = kl.ELU()  # kl.ReLU() or kl.ELU() or kl.Activation('swish')

    def __call__(self, inputs):
        x = inputs[0]
        input_length = x.shape[1]

        # Determine num_transformer_blocks
        if input_length >= 32000:
            self.num_transformer_blocks = 5
        elif input_length >= 16000:
            self.num_transformer_blocks = 4
        elif input_length >= 8000:
            self.num_transformer_blocks = 3
        elif input_length >= 4000:
            self.num_transformer_blocks = 2
            
        # Define kernel regularizer
        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)

        # Convolutional block 1
        shortcut = x  # Save the input for the residual connection
        x = kl.Conv1D(filters=360, kernel_size=27, padding='same', kernel_initializer=self.init,
                       kernel_regularizer=kernel_regularizer)(x)
        x = self.act(x)
        x = kl.BatchNormalization()(x)
        x = kl.MaxPooling1D(pool_size=4, strides=4)(x)
        x = kl.Dropout(0.3)(x)

        # Adjust shortcut dimension
        shortcut = kl.MaxPooling1D(pool_size=4, strides=4)(shortcut)  # Downsample shortcut
        shortcut = kl.Conv1D(filters=360, kernel_size=1, kernel_initializer=self.init, padding='same')(shortcut)  # Match filter count

        # Residual connection
        x = kl.add([x, shortcut])  # Add residual connection

        # Convolutional block 2
        shortcut = x  # Save the input for the residual connection
        x = kl.Conv1D(filters=480, kernel_size=18, padding='same', kernel_initializer=self.init,
                       kernel_regularizer=kernel_regularizer)(x)
        x = self.act(x)
        x = kl.BatchNormalization()(x)
        x = kl.MaxPooling1D(pool_size=4, strides=4)(x)
        x = kl.Dropout(0.3)(x)

        # Adjust shortcut dimension
        shortcut = kl.MaxPooling1D(pool_size=4, strides=4)(shortcut)  # Downsample shortcut
        shortcut = kl.Conv1D(filters=480, kernel_size=1, kernel_initializer=self.init, padding='same')(shortcut)  # Match filter count

        # Residual connection
        x = kl.add([x, shortcut])  # Add residual connection

        # Start alternating CNN and Transformer blocks
        for _ in range(self.num_transformer_blocks):
            shortcut = x  # Save the input for the residual connection
            x = kl.Conv1D(filters=512, kernel_size=8, padding='same', kernel_initializer=self.init,
                           kernel_regularizer=kernel_regularizer)(x)
            x = self.act(x)
            x = kl.BatchNormalization()(x)
            x = kl.MaxPooling1D(pool_size=2, strides=2)(x)
            x = kl.Dropout(0.4)(x)

            # Adjust shortcut dimension
            shortcut = kl.MaxPooling1D(pool_size=2, strides=2)(shortcut)  # Downsample shortcut
            shortcut = kl.Conv1D(filters=512, kernel_size=1, kernel_initializer=self.init, padding='same')(shortcut)  # Match filter count

            # Residual connection
            x = kl.add([x, shortcut])  # Add residual connection

            # Reshape to pass into Transformer blocks
            seq_len = x.shape[1]  # Updated sequence length after convolutions
            x = kl.Reshape((seq_len, 512))(x)

            # Transformer block
            attn_output = kl.MultiHeadAttention(num_heads=self.num_heads, key_dim=32)(x, x)
            attn_output = kl.Dropout(0.3)(attn_output)
            attn_output = kl.LayerNormalization(epsilon=1e-6)(attn_output + x)

            # Feedforward layer
            ff_output = kl.Dense(self.ff_dim, activation="gelu")(attn_output)
            ff_output = kl.Dropout(0.4)(ff_output)
            ff_output = kl.LayerNormalization(epsilon=1e-6)(ff_output + attn_output)

            x = ff_output
            
        # Additional Transformer Blocks
        seq_len = x.shape[1]  # Updated sequence length after convolutions
        x = kl.Reshape((seq_len, 512))(x)
        attn_output = kl.MultiHeadAttention(num_heads=self.num_heads, key_dim=32)(x, x)
        attn_output = kl.Dropout(0.5)(attn_output)
        attn_output = kl.LayerNormalization(epsilon=1e-6)(attn_output + x)

        # Feedforward layer
        ff_output = kl.Dense(self.ff_dim, activation="gelu")(attn_output)
        ff_output = kl.Dropout(0.5)(ff_output)
        ff_output = kl.LayerNormalization(epsilon=1e-6)(ff_output + attn_output)
        x = ff_output

        # Flatten the transformer output
        x = kl.Flatten()(x)

        # Fully connected layers
        x = kl.Dense(4096, kernel_initializer=self.init, kernel_regularizer=kernel_regularizer)(x)
        x = self.act(x)
        x = kl.Dropout(0.5)(x)

        x = kl.Dense(self.nb_hidden, kernel_initializer=self.init, kernel_regularizer=kernel_regularizer)(x)
        x = self.act(x)
        x = kl.Dropout(0.5)(x)

        return self._build(inputs, x)
        
        
        
        
class TranCNNAltRes2L(DnaModel):
    def __init__(self, nb_hidden=512, num_heads=16, ff_dim=512, num_transformer_blocks=2, *args, **kwargs):
        super(TranCNNAltRes2L, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.init = "he_normal"  # 'glorot_uniform' or "he_normal"
        self.act = kl.ELU()  # kl.ReLU() or kl.ELU() or kl.Activation('swish')

    def __call__(self, inputs):
        x = inputs[0]
        input_length = x.shape[1]

        # Determine num_transformer_blocks
        if input_length >= 32000:
            self.num_transformer_blocks = 5
        elif input_length >= 16000:
            self.num_transformer_blocks = 4
        elif input_length >= 8000:
            self.num_transformer_blocks = 3
        elif input_length >= 4000:
            self.num_transformer_blocks = 2
            
        # Define kernel regularizer
        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)

        # Convolutional block 1
        shortcut = x  # Save the input for the residual connection
        x = kl.Conv1D(filters=360, kernel_size=27, padding='same', kernel_initializer=self.init,
                       kernel_regularizer=kernel_regularizer)(x)
        x = self.act(x)
        x = kl.BatchNormalization()(x)
        
        x = kl.Conv1D(filters=360, kernel_size=27, padding='same', kernel_initializer=self.init,
                       kernel_regularizer=kernel_regularizer)(x)
        x = self.act(x)
        x = kl.BatchNormalization()(x)
        x = kl.MaxPooling1D(pool_size=4, strides=4)(x)
        x = kl.Dropout(0.3)(x)

        # Adjust shortcut dimension
        shortcut = kl.MaxPooling1D(pool_size=4, strides=4)(shortcut)  # Downsample shortcut
        shortcut = kl.Conv1D(filters=360, kernel_size=1, kernel_initializer=self.init, padding='same')(shortcut)

        # Residual connection
        x = kl.add([x, shortcut])  # Add residual connection

        # Convolutional block 2
        shortcut = x  # Save the input for the residual connection
        x = kl.Conv1D(filters=480, kernel_size=18, padding='same', kernel_initializer=self.init,
                       kernel_regularizer=kernel_regularizer)(x)
        x = self.act(x)
        x = kl.BatchNormalization()(x)
        
        x = kl.Conv1D(filters=480, kernel_size=18, padding='same', kernel_initializer=self.init,
                       kernel_regularizer=kernel_regularizer)(x)
        x = self.act(x)
        x = kl.BatchNormalization()(x)
        x = kl.MaxPooling1D(pool_size=4, strides=4)(x)
        x = kl.Dropout(0.3)(x)

        # Adjust shortcut dimension
        shortcut = kl.MaxPooling1D(pool_size=4, strides=4)(shortcut)  # Downsample shortcut
        shortcut = kl.Conv1D(filters=480, kernel_size=1, kernel_initializer=self.init, padding='same')(shortcut)

        # Residual connection
        x = kl.add([x, shortcut])  # Add residual connection

        # Start alternating CNN and Transformer blocks
        for _ in range(self.num_transformer_blocks):
            # Convolutional block
            shortcut = x  # Save the input for the residual connection
            x = kl.Conv1D(filters=512, kernel_size=8, padding='same', kernel_initializer=self.init,
                           kernel_regularizer=kernel_regularizer)(x)
            x = self.act(x)
            x = kl.BatchNormalization()(x)
            
            x = kl.Conv1D(filters=512, kernel_size=8, padding='same', kernel_initializer=self.init,
                           kernel_regularizer=kernel_regularizer)(x)
            x = self.act(x)
            x = kl.BatchNormalization()(x)
            x = kl.MaxPooling1D(pool_size=2, strides=2)(x)
            x = kl.Dropout(0.4)(x)

            # Adjust shortcut dimension
            shortcut = kl.MaxPooling1D(pool_size=2, strides=2)(shortcut)  # Downsample shortcut
            shortcut = kl.Conv1D(filters=512, kernel_size=1, kernel_initializer=self.init, padding='same')(shortcut)

            # Residual connection
            x = kl.add([x, shortcut])  # Add residual connection

            # Reshape to pass into Transformer blocks
            seq_len = x.shape[1]  # Updated sequence length after convolutions
            x = kl.Reshape((seq_len, 512))(x)

            # Transformer block
            attn_output = kl.MultiHeadAttention(num_heads=self.num_heads, key_dim=32)(x, x)
            attn_output = kl.Dropout(0.3)(attn_output)
            attn_output = kl.LayerNormalization(epsilon=1e-6)(attn_output + x)

            # Feedforward layer
            ff_output = kl.Dense(self.ff_dim, activation="gelu")(attn_output)
            ff_output = kl.Dropout(0.4)(ff_output)
            ff_output = kl.LayerNormalization(epsilon=1e-6)(ff_output + attn_output)

            x = ff_output
            
        # Additional Transformer Blocks
        seq_len = x.shape[1]  # Updated sequence length after convolutions
        x = kl.Reshape((seq_len, 512))(x)
        attn_output = kl.MultiHeadAttention(num_heads=self.num_heads, key_dim=32)(x, x)
        attn_output = kl.Dropout(0.5)(attn_output)
        attn_output = kl.LayerNormalization(epsilon=1e-6)(attn_output + x)

        # Feedforward layer
        ff_output = kl.Dense(self.ff_dim, activation="gelu")(attn_output)
        ff_output = kl.Dropout(0.5)(ff_output)
        ff_output = kl.LayerNormalization(epsilon=1e-6)(ff_output + attn_output)
        x = ff_output

        # Flatten the transformer output
        x = kl.Flatten()(x)

        # Fully connected layers
        x = kl.Dense(4096, kernel_initializer=self.init, kernel_regularizer=kernel_regularizer)(x)
        x = self.act(x)
        x = kl.Dropout(0.5)(x)

        x = kl.Dense(self.nb_hidden, kernel_initializer=self.init, kernel_regularizer=kernel_regularizer)(x)
        x = self.act(x)
        x = kl.Dropout(0.5)(x)

        return self._build(inputs, x)





# Class to handle B-spline generation
class BSpline:
    def __init__(self, x, df=None, knots=None, degree=3, intercept=False):
        self.x = x
        self.df = df
        self.knots = knots
        self.degree = degree
        self.intercept = intercept

    def create_spline(self):
        order = self.degree + 1
        inner_knots = []
        if self.df is not None and self.knots is None:
            n_inner_knots = self.df - order + (1 - self.intercept)
            if n_inner_knots < 0:
                n_inner_knots = 0
                print("df was too small; using %d" % (order - (1 - self.intercept)))
            if n_inner_knots > 0:
                inner_knots = np.percentile(self.x, 100 * np.linspace(0, 1, n_inner_knots + 2)[1:-1])
        elif self.knots is not None:
            inner_knots = self.knots

        all_knots = np.concatenate(([np.min(self.x), np.max(self.x)] * order, inner_knots))
        all_knots.sort()

        n_basis = len(all_knots) - (self.degree + 1)
        basis = np.empty((self.x.shape[0], n_basis), dtype=float)

        for i in range(n_basis):
            coefs = np.zeros((n_basis,))
            coefs[i] = 1
            basis[:, i] = splev(self.x, (all_knots, coefs, self.degree))

        if not self.intercept:
            basis = basis[:, 1:]

        return basis


# Class to handle spline factory creation
class SplineFactory:
    def __init__(self, n, df, log=False):
        self.n = n
        self.df = df
        self.log = log

    def create_factory(self):
        if self.log:
            dist = np.array(np.arange(self.n) - self.n / 2.0)
            dist = np.log(np.abs(dist) + 1) * (2 * (dist > 0) - 1)
            n_knots = self.df - 4
            knots = np.linspace(np.min(dist), np.max(dist), n_knots + 2)[1:-1]
            return tf.convert_to_tensor(BSpline(dist, knots=knots, intercept=True).create_spline())  # dtype=tf.float32)
        else:
            dist = np.arange(self.n)
            return tf.convert_to_tensor(BSpline(dist, df=self.df, intercept=True).create_spline())  # dtype=tf.float32)



class BSplineTransformation(tf.keras.layers.Layer):
    def __init__(self, degrees_of_freedom, log=False, scaled=False):
        super(BSplineTransformation, self).__init__()
        self.degrees_of_freedom = degrees_of_freedom
        self.log = log
        self.scaled = scaled
        self.spline_tr = None  # Initialize spline transformation as None

    def build(self, input_shape):
        spatial_dim = input_shape[-1]
        # Don't create the spline matrix yet, delay until call()
        self.input_shape_dim = spatial_dim

    @tf.function
    def call(self, inputs):
        # Only create spline_tr when call() is executed for the first time
        if self.spline_tr is None:
            self.spline_tr = SplineFactory(self.input_shape_dim, self.degrees_of_freedom, log=self.log).create_factory()
            if self.scaled:
                self.spline_tr = self.spline_tr / float(self.input_shape_dim)

        # Now run the matmul operation
        return tf.matmul(inputs, tf.cast(self.spline_tr, inputs.dtype))




class Sei(DnaModel):
    def __init__(self, nb_hidden=512, *args, **kwargs):
        super(Sei, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

        kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)

        # First convolutional block
        self.lconv1 = tf.keras.Sequential([
            kl.Conv1D(480, kernel_size=9, padding='same', kernel_initializer=self.init, kernel_regularizer=kernel_regularizer),
            kl.Conv1D(480, kernel_size=9, padding='same', kernel_initializer=self.init, kernel_regularizer=kernel_regularizer),
        ])

        self.conv1 = tf.keras.Sequential([
            kl.Conv1D(480, kernel_size=9, padding='same', kernel_initializer=self.init, kernel_regularizer=kernel_regularizer),
            kl.Activation('relu'),
            kl.Conv1D(480, kernel_size=9, padding='same', kernel_initializer=self.init, kernel_regularizer=kernel_regularizer),
            kl.Activation('relu'),
        ])

        # Second convolutional block
        self.lconv2 = tf.keras.Sequential([
            kl.MaxPooling1D(pool_size=4, strides=4),
            kl.Dropout(0.2),
            kl.Conv1D(640, kernel_size=9, padding='same', kernel_initializer=self.init, kernel_regularizer=kernel_regularizer),
            kl.Conv1D(640, kernel_size=9, padding='same', kernel_initializer=self.init, kernel_regularizer=kernel_regularizer),
        ])

        self.conv2 = tf.keras.Sequential([
            kl.Dropout(0.2),
            kl.Conv1D(640, kernel_size=9, padding='same', kernel_initializer=self.init, kernel_regularizer=kernel_regularizer),
            kl.Activation('relu'),
            kl.Conv1D(640, kernel_size=9, padding='same', kernel_initializer=self.init, kernel_regularizer=kernel_regularizer),
            kl.Activation('relu'),
        ])

        # Third convolutional block
        self.lconv3 = tf.keras.Sequential([
            kl.MaxPooling1D(pool_size=4, strides=4),
            kl.Dropout(0.2),
            kl.Conv1D(960, kernel_size=9, padding='same', kernel_initializer=self.init, kernel_regularizer=kernel_regularizer),
            kl.Conv1D(960, kernel_size=9, padding='same', kernel_initializer=self.init, kernel_regularizer=kernel_regularizer),
        ])

        self.conv3 = tf.keras.Sequential([
            kl.Dropout(0.2),
            kl.Conv1D(960, kernel_size=9, padding='same', kernel_initializer=self.init, kernel_regularizer=kernel_regularizer),
            kl.Activation('relu'),
            kl.Conv1D(960, kernel_size=9, padding='same', kernel_initializer=self.init, kernel_regularizer=kernel_regularizer),
            kl.Activation('relu'),
        ])

        # Dilated convolutions
        self.dconv1 = tf.keras.Sequential([
            kl.Dropout(0.10),
            kl.Conv1D(960, kernel_size=5, dilation_rate=2, padding='same', kernel_initializer=self.init, kernel_regularizer=kernel_regularizer),
            kl.Activation('relu'),
        ])
        self.dconv2 = tf.keras.Sequential([
            kl.Dropout(0.10),
            kl.Conv1D(960, kernel_size=5, dilation_rate=4, padding='same', kernel_initializer=self.init, kernel_regularizer=kernel_regularizer),
            kl.Activation('relu'),
        ])
        self.dconv3 = tf.keras.Sequential([
            kl.Dropout(0.10),
            kl.Conv1D(960, kernel_size=5, dilation_rate=8, padding='same', kernel_initializer=self.init, kernel_regularizer=kernel_regularizer),
            kl.Activation('relu'),
        ])
        self.dconv4 = tf.keras.Sequential([
            kl.Dropout(0.10),
            kl.Conv1D(960, kernel_size=5, dilation_rate=16, padding='same', kernel_initializer=self.init, kernel_regularizer=kernel_regularizer),
            kl.Activation('relu'),
        ])
        self.dconv5 = tf.keras.Sequential([
            kl.Dropout(0.10),
            kl.Conv1D(960, kernel_size=5, dilation_rate=25, padding='same', kernel_initializer=self.init, kernel_regularizer=kernel_regularizer),
            kl.Activation('relu'),
        ])

        self._spline_df = int(128 / 8)
        self.spline_tr = tf.keras.Sequential([
            kl.Dropout(0.5),
            BSplineTransformation(self._spline_df, scaled=False)
        ])

        # Dense layer
        self.fc1 = tf.keras.Sequential([
            kl.Dense(nb_hidden, kernel_initializer=self.init, kernel_regularizer=kernel_regularizer),
            kl.Activation('relu'),
        ])


    def __call__(self, inputs):
        x = inputs[0]
        lout1 = self.lconv1(x)
        out1 = self.conv1(lout1)

        lout2 = self.lconv2(out1 + lout1)
        out2 = self.conv2(lout2)

        lout3 = self.lconv3(out2 + lout2)
        out3 = self.conv3(lout3)

        dconv_out1 = self.dconv1(out3 + lout3)
        cat_out1 = out3 + dconv_out1
        dconv_out2 = self.dconv2(cat_out1)
        cat_out2 = cat_out1 + dconv_out2
        dconv_out3 = self.dconv3(cat_out2)
        cat_out3 = cat_out2 + dconv_out3
        dconv_out4 = self.dconv4(cat_out3)
        cat_out4 = cat_out3 + dconv_out4
        dconv_out5 = self.dconv5(cat_out4)
        out = cat_out4 + dconv_out5
 
        spline_out = self.spline_tr(out)
        reshape_out = tf.reshape(spline_out, [spline_out.shape[0], -1])
        fc1_out = self.fc1(reshape_out)

        return self._build(inputs,fc1_out)






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
