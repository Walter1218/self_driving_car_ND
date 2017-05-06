# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from scipy.ndimage import rotate
from scipy.stats import bernoulli
import cv2,os
import scipy.misc
import cv2
#PATH = './data_extra'
PATH = './data/'
data_csv = '/driving_log.csv'
DRIVING_LOG_FILE = ''#'./data_extra/driving_log.csv'
IMG_PATH = ''#'./data_extra/'
STEERING_COEFFICIENT = 0.229
# read the csv file
def read_csv():
    #training_data = pd.read_csv(PATH+data_csv,names=['center','left','right','steering','throttle','break','speed'])
    training_data = pd.read_csv(PATH+data_csv,names=None)
    print(training_data.shape)
    training_data[['center','left','right']]
    X = training_data[['center','left','right']]
    Y = training_data['steering']
    #print(X.head())
    #X_left  = X['left'].as_matrix()
    #X_right = X_['right'].as_matrix()
    #X_center = X['center'].as_matrix()
    #Y = Y.as_matrix()
    #return X_left, X_right, X_center, Y
    return X, Y
    
def datashuffle(X, Y):
    from sklearn.utils import shuffle
    X, Y = shuffle(X, Y)
    return X, Y

def trainval_split(X, Y):
    from sklearn.model_selection import train_test_split
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
    #X_Tleft  = X_train['left'].as_matrix()
    #X_Tright = X_train['right'].as_matrix()
    #X_Tcenter = X_train['center'].as_matrix()
    #Y_T = Y_train.as_matrix()
    
    #X_Vleft  = X_val['left'].as_matrix()
    #X_Vright = X_val['right'].as_matrix()
    #X_Vcenter = X_val['center'].as_matrix()
    #Y_V = Y_val.as_matrix()
    #return X_Tleft, X_Tright, X_Tcenter,Y_T, X_Vleft, X_Vright, X_Vcenter, Y_V
    return X_train, X_val, Y_train, Y_val

def random_flip(img, label):
    choice = np.random.choice([0,1])
    if choice == 1:
        img, label = cv2.flip(img, 1), -label
    return (img, label)

def random_brightness(img, label):
    br_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    coin = np.random.randint(2)
    if coin == 0:
        random_bright = 0.2 + np.random.uniform(0.2, 0.6)
        br_img[:, :, 2] = br_img[:, :, 2] * random_bright
    br_img = cv2.cvtColor(br_img, cv2.COLOR_HSV2RGB)
    return br_img, label

def random_view(index, X_left, X_right, X_center, Y):
    PATH = './data/'
    choice = np.random.choice([0,1,2])
    y = Y[index]
    if choice == 0:
        image = plt.imread(PATH+'/'+X_left[index].strip())
        dsteering = y+ 0.25
        return image,dsteering
    elif choice == 2:
        image = plt.imread(PATH+'/'+X_right[index].strip())
        dsteering = y- 0.25
        return image,dsteering
    else:
        image = plt.imread(PATH+'/'+X_center[index].strip())
        dsteering = y
        return image,dsteering
    
def generate_train(X_center,X_left,X_right,Y):
    """
    data augmentation
    transformed image & crop
    """
    index = np.random.randint(0,len(Y))
    img,y = random_view(index, X_left, X_right, X_center, Y)
    img,y = random_brightness(img, y)
    img, y = random_flip(img, y)
    return img, y
    #num = np.random.randint(0, len(Y))
    #img, y = 
    #img = cv2.resize(X[num,:,:,:], (64,64), cv2.INTER_AREA)
    #img = crop(X[num,:,:,:], 64, 64)
    #img = X[num,:,:,:]
    #img_, y_ = flip(img, Y[num])
    #img_ = brightness(img)
    #img_ = random_noise(img_)
    #return img, Y[num]

def generate_train_batch(center, left, right, steering, batch_size):
    """ compose training batch set """
    image_set = np.zeros((batch_size, 160, 320, 3))
    steering_set = np.zeros(batch_size)

    while 1:
        for i in range(batch_size):
            img, steer = generate_train(center,left, right, steering)
            image_set[i] = img
            steering_set[i] = steer
        yield image_set, steering_set
