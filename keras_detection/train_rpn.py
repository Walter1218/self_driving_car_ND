"""
The model seems can be fitting, but we need add fast-rnn classification model at the end of rpn layer, and see the performance of this model
"""
import pandas as pd
import numpy as np
import cv2, utils
import random
import copy
import threading,batch_generate,netarch
import itertools,sys
import numpy.random as npr
import tensorflow as tf
import keras, losses
from keras import backend as K
from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model
data = pd.read_csv('voc.csv')
data = data.drop('Unnamed: 0', 1)
data['File_Path'] = './VOCdevkit2007/VOC2007/JPEGImages/' + data['Frame']
print(data.head())
gen = batch_generate.batch_generate(data, batch_size=1)
input_shape_img = (None, None, 3)
img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(300, 4))

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = netarch.base_net(img_input, trainable=False)

# define the RPN, built on the base layers
num_anchors = 9
rpn = netarch.rpn(shared_layers, num_anchors)
model_rpn = Model(img_input, rpn[:2])
model_rpn.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels.h5', by_name=True)
optimizer = Adam(lr=1e-5)
model_rpn.compile(optimizer=optimizer, loss=[losses.rpn_classification(num_anchors), losses.rpn_regression(num_anchors)])
epoch_length = 1000
num_epochs = 4
iter_num = 0
epoch_num = 0
while True:
    try:
        X, Y = next(gen)
        loss_rpn = model_rpn.train_on_batch(X, Y)
        P_rpn = model_rpn.predict_on_batch(X)
        iter_num += 1
        print('iter {0}, epoch {1}, loss {2}'.format(iter_num, epoch_num, loss_rpn))
        if iter_num == epoch_length:
            iter_num = 0
            epoch_num += 1
        if epoch_num == num_epochs:
            print('Training complete, exiting.')
            model_rpn.save_weights('rpn.h5')
            sys.exit()
    except Exception as e:
        print('Exception: {}'.format(e))
        continue
