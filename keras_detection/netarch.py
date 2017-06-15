from __future__ import print_function
from __future__ import absolute_import

from keras.layers import Input, Dense, Activation, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D, TimeDistributed, convolutional, core
from roipool import RoiPoolingConv
from keras import backend as K
import tensorflow as tf
def base_net(input_tensor=None, trainable=False):
    """
    basical architecture of vgg16(conv1 ----conv5-3)
    """
    input_shape = (None, None, 3)
    img_input = Input(tensor=input_tensor, shape=input_shape)
    conv1 = convolutional.Conv2D(64, 3, 3, activation='relu',border_mode='same',name='conv1_1', trainable = trainable)(img_input)
    conv1 = convolutional.Conv2D(64, 3, 3, activation='relu',border_mode='same',name='conv1_2', trainable = trainable)(conv1)
    pool1 = MaxPooling2D(strides=(2, 2))(conv1)

    conv2 = convolutional.Conv2D(128, 3, 3, activation='relu',border_mode='same',name='conv2_1', trainable = trainable)(pool1)
    conv2 = convolutional.Conv2D(128, 3, 3, activation='relu',border_mode='same',name='conv2_2', trainable = trainable)(conv2)
    pool2 = MaxPooling2D(strides=(2, 2))(conv2)

    conv3 = convolutional.Conv2D(256, 3, 3, activation='relu',border_mode='same',name='conv3_1', trainable = trainable)(pool2)
    conv3 = convolutional.Conv2D(256, 3, 3, activation='relu',border_mode='same',name='conv3_2', trainable = trainable)(conv3)
    conv3 = convolutional.Conv2D(256, 3, 3, activation='relu',border_mode='same',name='conv3_3', trainable = trainable)(conv3)
    pool3 = MaxPooling2D(strides=(2, 2))(conv3)

    conv4 = convolutional.Conv2D(512, 3, 3, activation='relu',border_mode='same',name='conv4_1', trainable = trainable)(pool3)
    conv4 = convolutional.Conv2D(512, 3, 3, activation='relu',border_mode='same',name='conv4_2', trainable = trainable)(conv4)
    conv4 = convolutional.Conv2D(512, 3, 3, activation='relu',border_mode='same',name='conv4_3', trainable = trainable)(conv4)
    pool4 = MaxPooling2D(strides=(2, 2))(conv4)

    conv5 = convolutional.Conv2D(512, 3, 3, activation='relu',border_mode='same',name='conv5_1', trainable = trainable)(pool4)
    conv5 = convolutional.Conv2D(512, 3, 3, activation='relu',border_mode='same',name='conv5_2', trainable = trainable)(conv5)
    conv5 = convolutional.Conv2D(512, 3, 3, activation='relu',border_mode='same',name='conv5_3', trainable = trainable)(conv5)

    return conv5

def rpn(base_layers,num_anchors):

    x = convolutional.Conv2D(512, 3, 3, border_mode='same', activation='relu',name='rpn_conv1')(base_layers)
    x = convolutional.Conv2D(1024, 3, 3, border_mode='same', activation='relu',name='rpn_conv2')(x)
    x = convolutional.Conv2D(512, 3, 3, border_mode='same', activation='relu',name='rpn_conv3')(x)

    x_class = convolutional.Conv2D(num_anchors, 1, 1, activation='sigmoid', border_mode='same', name='rpn_out_class')(x)

    x_regr = convolutional.Conv2D(num_anchors * 4, 1, 1, activation=None, name='rpn_out_regress')(x)

    return [x_class, x_regr, base_layers]

def classification(x_tensor, input_roi, limited_roi, nb_classes = 21, trainable=False):
    pooling_regions = 14
    input_shape = (limited_roi,14,14,512)
    #pool = ROI(14, input_roi)(x_tensor)
    pool = RoiPoolingConv(pooling_regions, limited_roi)([x_tensor, input_roi])

    flatten = TimeDistributed(Flatten())(pool)
    fc6 = TimeDistributed(Dense(4096, activation = 'relu'), name = 'fc6')(flatten)
    fc7 = TimeDistributed(Dense(4096, activation = 'relu'), name = 'fc7')(fc6)
    #score = Dense(nb_classes)(fc7)
    #score = Activation('softmax')(score)

    score = TimeDistributed(Dense(nb_classes, activation = 'softmax'), name = 'cls_score')(fc7)
    #boxes = Dense(4 * (nb_classes - 1))(fc7)
    #boxes = Activation('linear')(boxes)

    boxes = TimeDistributed(Dense(4 * (nb_classes - 1), activation = 'linear'), name = 'bbox')(fc7)
    return [score, boxes]
