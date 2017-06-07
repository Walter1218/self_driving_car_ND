from __future__ import print_function
from __future__ import absolute_import

from keras.layers import Input, Add, Dense, Activation, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D, TimeDistributed

from keras import backend as K
def base_net(input_tensor=None, trainable=False):
    """
    basical architecture of vgg16(conv1 ----conv5-3)
    """
    input_shape = (None, None, 3)
    img_input = Input(shape=input_shape)
    conv1 = Convolution2D(64, (3, 3), strides=(2, 2), name='conv1_1', trainable = trainable)(img_input)
