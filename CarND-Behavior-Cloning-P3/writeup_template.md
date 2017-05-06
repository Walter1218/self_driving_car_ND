#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.ipynb containing the script to create and train the model
* util.py containing the script to trainval split, and data augmentation code.
* drive.py for driving the car in autonomous mode
* model.h5 and model.json containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.json
```

####3. Submission code is usable and readable

The model.ipynb file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.(more details please see the model.ipynb)

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 10 to 96  

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer .

The final model only have 1.1mb size, and use lots of 1*1 convolution instead fully connencted layer.

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting, also I use batch normalization here. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road .  I labeled left and right images as center images here.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to use NVIDIA's SDC model as basical model, and the change it to other architecture,

My first step was to use a convolution neural network model similar to the NVIDIA architecture, I thought this model might be appropriate because this model has been tested by NIVIDA in real road. The model is working well in the simulation, but the model size and the parameter is very large, so I change the convolution filter numbers, and the structure of the network, I only use one fully connected layer here, and only have 10 hidden numbers, also I add batchnormalization and dropout to avoid overfitting. 

The final step was to run the simulator to see how well the car was driving around track one. and the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes ...

____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_2 (Lambda)                (None, 160, 320, 3)   0           lambda_input_2[0][0]             
____________________________________________________________________________________________________
batchnormalization_2 (BatchNorma (None, 160, 320, 3)   640         lambda_2[0][0]                   
____________________________________________________________________________________________________
convolution2d_11 (Convolution2D) (None, 80, 160, 3)    12          batchnormalization_2[0][0]       
____________________________________________________________________________________________________
maxpooling2d_7 (MaxPooling2D)    (None, 79, 159, 3)    0           convolution2d_11[0][0]           
____________________________________________________________________________________________________
cropping2d_2 (Cropping2D)        (None, 49, 159, 3)    0           maxpooling2d_7[0][0]             
____________________________________________________________________________________________________
convolution2d_12 (Convolution2D) (None, 25, 80, 36)    1008        cropping2d_2[0][0]               
____________________________________________________________________________________________________
activation_11 (Activation)       (None, 25, 80, 36)    0           convolution2d_12[0][0]           
____________________________________________________________________________________________________
maxpooling2d_8 (MaxPooling2D)    (None, 24, 79, 36)    0           activation_11[0][0]              
____________________________________________________________________________________________________
convolution2d_13 (Convolution2D) (None, 12, 40, 48)    15600       maxpooling2d_8[0][0]             
____________________________________________________________________________________________________
activation_12 (Activation)       (None, 12, 40, 48)    0           convolution2d_13[0][0]           
____________________________________________________________________________________________________
maxpooling2d_9 (MaxPooling2D)    (None, 11, 39, 48)    0           activation_12[0][0]              
____________________________________________________________________________________________________
convolution2d_14 (Convolution2D) (None, 6, 20, 48)     20784       maxpooling2d_9[0][0]             
____________________________________________________________________________________________________
activation_13 (Activation)       (None, 6, 20, 48)     0           convolution2d_14[0][0]           
____________________________________________________________________________________________________
dropout_4 (Dropout)              (None, 6, 20, 48)     0           activation_13[0][0]              
____________________________________________________________________________________________________
maxpooling2d_10 (MaxPooling2D)   (None, 5, 19, 48)     0           dropout_4[0][0]                  
____________________________________________________________________________________________________
convolution2d_15 (Convolution2D) (None, 5, 19, 64)     27712       maxpooling2d_10[0][0]            
____________________________________________________________________________________________________
activation_14 (Activation)       (None, 5, 19, 64)     0           convolution2d_15[0][0]           
____________________________________________________________________________________________________
dropout_5 (Dropout)              (None, 5, 19, 64)     0           activation_14[0][0]              
____________________________________________________________________________________________________
maxpooling2d_11 (MaxPooling2D)   (None, 4, 18, 64)     0           dropout_5[0][0]                  
____________________________________________________________________________________________________
convolution2d_16 (Convolution2D) (None, 4, 18, 64)     36928       maxpooling2d_11[0][0]            
____________________________________________________________________________________________________
activation_15 (Activation)       (None, 4, 18, 64)     0           convolution2d_16[0][0]           
____________________________________________________________________________________________________
dropout_6 (Dropout)              (None, 4, 18, 64)     0           activation_15[0][0]              
____________________________________________________________________________________________________
maxpooling2d_12 (MaxPooling2D)   (None, 3, 17, 64)     0           dropout_6[0][0]                  
____________________________________________________________________________________________________
convolution2d_17 (Convolution2D) (None, 3, 17, 96)     55392       maxpooling2d_12[0][0]            
____________________________________________________________________________________________________
activation_16 (Activation)       (None, 3, 17, 96)     0           convolution2d_17[0][0]           
____________________________________________________________________________________________________
convolution2d_18 (Convolution2D) (None, 3, 17, 96)     83040       activation_16[0][0]              
____________________________________________________________________________________________________
activation_17 (Activation)       (None, 3, 17, 96)     0           convolution2d_18[0][0]           
____________________________________________________________________________________________________
convolution2d_19 (Convolution2D) (None, 3, 17, 50)     4850        activation_17[0][0]              
____________________________________________________________________________________________________
activation_18 (Activation)       (None, 3, 17, 50)     0           convolution2d_19[0][0]           
____________________________________________________________________________________________________
convolution2d_20 (Convolution2D) (None, 3, 17, 10)     510         activation_18[0][0]              
____________________________________________________________________________________________________
activation_19 (Activation)       (None, 3, 17, 10)     0           convolution2d_20[0][0]           
____________________________________________________________________________________________________
flatten_2 (Flatten)              (None, 510)           0           activation_19[0][0]              
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            5110        flatten_2[0][0]                  
____________________________________________________________________________________________________
activation_20 (Activation)       (None, 10)            0           dense_3[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          activation_20[0][0]              

Total params: 251,597
Trainable params: 251,277
Non-trainable params: 320

####3. Creation of the Training Set & Training Process

My training data is based on the Udacity's original dataset, and I use some dataaugmentation technology to produce more data, also I try to collect my own data from the simulation, and combined them together to generate my final training data and validate data.

For the data augmentation technology, I applied flip, brightness and random view generate. I also try to add random gaussian noise to those dataset, But the results is not good, so I removed this step. And I use keras data batch generator to produce the training data for each mini batch. 
