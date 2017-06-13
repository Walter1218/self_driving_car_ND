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
#data = data[(data['label'] == 0)].reset_index()
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
#classifier = netarch.classification(shared_layers, roi_input, 300, nb_classes=21, trainable=True)
#model_classifier = Model([img_input, roi_input], classifier)
#model_all = Model([img_input, roi_input], rpn[:2] + classifier)
model_rpn.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels.h5', by_name=True)
#model_classifier.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels.h5', by_name=True)
optimizer = Adam(lr=1e-5)
#optimizer_classifier = Adam(lr=1e-5)
model_rpn.compile(optimizer=optimizer, loss=[losses.rpn_classification(num_anchors), losses.rpn_regression(num_anchors)])
#model_classifier.compile(optimizer=optimizer_classifier, loss=[losses.classification, losses.regression(len(classes_count)-1)], metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})
#model_all.compile(optimizer='sgd', loss='mae')

epoch_length = 10000
num_epochs = 20
iter_num = 0
epoch_num = 0

losses = np.zeros((epoch_length, 5))
rpn_accuracy_rpn_monitor = []
rpn_accuracy_for_epoch = []

best_loss = np.Inf
while True:
    #try:
    X, Y, gta = next(gen)
    loss_rpn = model_rpn.train_on_batch(X, Y)
    #it will return cls, regression bbox, and base feature map
    P_rpn = model_rpn.predict_on_batch(X)
    #The input structure is [boxes, scores, maximum]
    #roi = utils.propose(P_rpn[1], P_rpn[0], 300)
    rois, scores = utils.propose_cpu(P_rpn[1], P_rpn[0], 300)
    #utils.propose_cpu(P_rpn[1], P_rpn[0], 300)
    #print(rois.shape)
    utils.cal_accuracy(gta, rois, scores)
    #print(np.asarray(roi))
    #print(roi[0])
    #print(P_rpn[0], P_rpn[1])
    iter_num += 1
    print('iter {0}, epoch {1}, loss {2}'.format(iter_num, epoch_num, loss_rpn))
    if iter_num == epoch_length:
        iter_num = 0
        epoch_num += 1
    if epoch_num == num_epochs:
        print('Training complete, exiting.')
        model_rpn.save_weights('rpn.h5')
        sys.exit()
    #except Exception as e:
    #    print('Exception: {}'.format(e))
    #    continue
