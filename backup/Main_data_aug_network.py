#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import tensorflow as tf
import random as rn
import os
import sys
from functools import reduce
from tensorflow.keras.optimizers import SGD
from sklearn.linear_model import LogisticRegression

import pandas as pd
import pyarrow.parquet as pq
import csv
import glob
from keras import models
from keras import layers
import keras 
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from collections import Counter
from keras.callbacks import History
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline      # for warping
from transforms3d.axangles import axangle2mat  # for rotation

import xlsxwriter
np.set_printoptions(threshold=sys.maxsize)


# In[12]:


# ## 1. Jittering

# #### Hyperparameters :  sigma = standard devitation (STD) of the noise
def DA_Jitter(X, sigma=0.01):
    myNoise = np.random.normal(loc=0, scale=sigma, size=X.shape)
    return X + myNoise


# ## 2. Scaling

# #### Hyperparameters :  sigma = STD of the zoom-in/out factor
def DA_Scaling(X, sigma=0.5):
    scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1, X.shape[1]))  # shape=(1,3)
    myNoise = np.matmul(np.ones((X.shape[0], 1)), scalingFactor)
    return X * myNoise


# ## 3. Magnitude Warping

# #### Hyperparameters :  sigma = STD of the random knots for generating curves
#
# #### knot = # of knots for the random curves (complexity of the curves)

# "Scaling" can be considered as "applying constant noise to the entire samples" whereas "Jittering" can be considered as "applying different noise to each sample".

# "Magnitude Warping" can be considered as "applying smoothly-varing noise to the entire samples"


## This example using cubic splice is not the best approach to generate random curves.
## You can use other aprroaches, e.g., Gaussian process regression, Bezier curve, etc.
def GenerateRandomCurves(X, sigma=0.2, knot=4):
    xx = (np.ones((X.shape[1], 1)) * (np.arange(0, X.shape[0], (X.shape[0] - 1) / (knot + 1)))).transpose()
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2, X.shape[1]))
    x_range = np.arange(X.shape[0])
    random_curves = []
    for i in range(X.shape[-1]):
        cs = CubicSpline(xx[:, i], yy[:, i])
        random_curves.append(cs(x_range))
    return np.array(random_curves).transpose()


def DA_MagWarp(X, sigma=0.2):
    return X * GenerateRandomCurves(X, sigma)


# ## 4. Time Warping

# #### Hyperparameters :  sigma = STD of the random knots for generating curves
#
# #### knot = # of knots for the random curves (complexity of the curves)

def DistortTimesteps(X, sigma=0.2):
    tt = GenerateRandomCurves(X, sigma)  # Regard these samples aroun 1 as time intervals
    tt_cum = np.cumsum(tt, axis=0)  # Add intervals to make a cumulative graph
    # Make the last value to have X.shape[0]
    for i in range(X.shape[-1]):
        t_scale = (X.shape[0] - 1) / tt_cum[-1, i]
        tt_cum[:, i] = tt_cum[:, i] * t_scale
    return tt_cum


def DA_TimeWarp(X, sigma=0.2):
    tt_new = DistortTimesteps(X, sigma)
    X_new = np.zeros(X.shape)
    x_range = np.arange(X.shape[0])
    for i in range(X.shape[-1]):
        X_new[:, i] = np.interp(x_range, tt_new[:, i], X[:, i])
    return X_new


# ## 5. Rotation

# #### Hyperparameters :  N/A

def DA_Rotation(X):
#     axis = np.random.uniform(low=-1, high=1, size=X.shape[1])
#     print(axis)
    angle = np.random.uniform(low=-np.pi, high=np.pi)
#     axis = np.array([axangle2mat([0,1,0], angle)] * X.shape[1])
#     print(axis.shape)

#     return X * (axangle2mat([0,1,0], angle)) 
    return X
#                 np.matmul(X, axis)


# ## 6. Permutation

# #### Hyperparameters :  nPerm = # of segments to permute
# #### minSegLength = allowable minimum length for each segment

def DA_Permutation(X, nPerm=4, minSegLength=10):
    X_new = np.zeros(X.shape)
    idx = np.random.permutation(nPerm)
    bWhile = True
    while bWhile:
        segs = np.zeros(nPerm + 1, dtype=int)
        segs[1:-1] = np.sort(np.random.randint(minSegLength, X.shape[0] - minSegLength, nPerm - 1))
        segs[-1] = X.shape[0]
        if np.min(segs[1:] - segs[0:-1]) > minSegLength:
            bWhile = False
    pp = 0
    for ii in range(nPerm):
        x_temp = X[segs[idx[ii]]:segs[idx[ii] + 1], :]
        X_new[pp:pp + len(x_temp), :] = x_temp
        pp += len(x_temp)
    return X_new


# In[13]:


def augment_data(train_set, train_label, function):


    train_set_original = train_set
#     START AUGMENTING
    ARR,LABEL = [], []
    
    # select random indices
    number_of_rows = int(train_set_original.shape[1] * 0.8)

#     random indices has to be the same for every dimension so that the label can be accurate
    random_indices = np.sort(np.random.choice(train_set_original.shape[1]-1, size=int(number_of_rows), replace=False))
#     print(random_indices)
    #     iterate over every dimension
#     augment every dimension of the tensor so that at the end we have a tensor augmented
    for train_set_one in train_set_original:
    # take partial array with random indices
        train_set_one = train_set_one[random_indices,:]
    # perform jittering on the partial array
        train_set_one = train_set_one.transpose()
#         SCALE AUGMENT
        if function == 'scale':
            train_set_one = DA_Scaling(train_set_one)
        elif function == 'jitter':
            train_set_one = DA_Jitter(train_set_one)
        elif function == 'magWarp':
            train_set_one = DA_MagWarp(train_set_one)
        elif function == 'timeWarp':
            train_set_one = DA_TimeWarp(train_set_one)
        elif function == 'rotation':
            train_set_one = DA_Rotation(train_set_one)
        elif function == 'permutation':
            train_set_one = DA_Permutation(train_set_one)
        else:
            print("Error no augmentation function")
            break
            
        train_set_one = train_set_one.transpose()
        
#     create an array ARR only of augmented data
        ARR = [*ARR, train_set_one]
    
    ARR = np.array(ARR)
    # take the label and add them as the label for the new augmented data
    LABEL = np.array(train_label[random_indices])
#     print(LABEL)
#     we have ARR which is of shape (6, row, col) with the augmented data
#     and train_set which is of shape (6, row, col) with the non augmented data
    
    train_set_augmented = np.concatenate((train_set, ARR), axis = 1)
    print(train_set[0,0,0])
    print(ARR[0,0,0])
    print(train_set[1,0,0])
    print(ARR[1,0,0])
    print(train_set[2,0,0])
    print(ARR[2,0,0])
    print(train_set[3,0,0])
    print(ARR[3,0,0])
    train_label = np.array(train_label)
    label_set_augmented = np.concatenate((train_label, LABEL))
    
    return train_set_augmented, label_set_augmented


# In[14]:


def model_network():
    seed_value = 34567892
    os.environ['PYTHONHASHSEED']=str(seed_value)
    tf.random.set_seed(seed_value)
        
#     callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=(2400,)))
    model.add(Dense(1, activation='sigmoid'))
    opt = Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


# In[17]:


# # for each technique run tec_len times the model to obtain an average of accuracy and fill the csv table with results
techniques = ['jitter', 'scale', 'magWarp', 'timeWarp', 'rotation', 'permutation']
# techniques = ['scale']
# timeWarp change not that much the data
arr_eda, arr_bvc,arr_acc,arr_tem = [],[],[],[]
arr_eda_percentage, arr_bvc_percentage,arr_acc_percentage,arr_tem_percentage = [],[],[],[]
tec_eda, tec_bvc, tec_acc, tec_tem = 0,0,0,0

train_set = np.load('train_set_original.npy',  encoding='ASCII')
train_label = np.load('train_label_original.npy',  encoding='ASCII')
test_set = np.load('test_set.npy',  encoding='ASCII')
test_label = np.load('test_label.npy',  encoding='ASCII')
# print(train_label)

train_label = train_label.reshape(train_label.shape[0], 1)
test_label = test_label.reshape(test_label.shape[0], 1)

tec_eda, tec_bvc, tec_acc, tec_tem = 0,0,0,0
for dim in range(0, 4):
    train_set_arr = train_set[dim]
    test_set_arr = test_set[dim]

    model = model_network()

    #       ORIGINAL SET
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    histor = model.fit(train_set_arr, train_label, epochs=25, batch_size= 128, shuffle=False, verbose=0, callbacks = [callback])
    scores1 = model.evaluate(test_set_arr, test_label, verbose=0)
    print("evaluate original: ",scores1[1]/1)
    if dim == 0:
        tec_eda = scores1[1]
    elif dim == 1:
        tec_bvc = scores1[1]
    elif dim == 2:
        tec_acc = scores1[1]
    elif dim == 3:
        tec_tem = scores1[1]
            

    #     --------------------------------------------------------------------


for technique in techniques:
    print(technique)
    avg_tec_eda, avg_tec_bvc, avg_tec_acc, avg_tec_tem = 0,0,0,0
    tec_len = 10
    
# loop tec_len times to get the average of a tecnique
    for avg_t in range(0, tec_len):
        train_set = np.load('train_set_original.npy',  encoding='ASCII')
        train_label = np.load('train_label_original.npy',  encoding='ASCII')
        test_set = np.load('test_set.npy',  encoding='ASCII')
        test_label = np.load('test_label.npy',  encoding='ASCII')
        train_label = train_label.reshape(train_label.shape[0], 1)
        test_label = test_label.reshape(test_label.shape[0], 1)

        train_set_augmented, label_set_augmented = augment_data(train_set, train_label, technique)
        
        
        for dim in range(0, 4):
            train_set_arr_augment = train_set_augmented[dim]
            test_set_arr = test_set[dim]
            model = model_network()

            #       AUGMENTATION
            callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
            histor = model.fit(train_set_arr_augment, label_set_augmented, epochs=25, batch_size= 128, shuffle=True,verbose=0, callbacks = [callback])
            scores2 = model.evaluate(test_set_arr, test_label,verbose=0)
            #     --------------------------------------------------------------------
            
            print("evaluate augmented: ",scores2[1]/1)


            if dim == 0:
                avg_tec_eda += scores2[1]
            elif dim == 1:
                avg_tec_bvc += scores2[1]
            elif dim == 2:
                avg_tec_acc += scores2[1]
            elif dim == 3:
                avg_tec_tem += scores2[1]
            

    arr_eda.append(avg_tec_eda/tec_len)
    arr_bvc.append(avg_tec_bvc/tec_len)
    arr_acc.append(avg_tec_acc/tec_len)
    arr_tem.append(avg_tec_tem/tec_len)
    
    arr_eda_percentage.append(round((avg_tec_eda/tec_len - tec_eda) * 100, 0))
    arr_bvc_percentage.append(round((avg_tec_bvc/tec_len - tec_bvc) * 100, 0))
    arr_acc_percentage.append(round((avg_tec_acc/tec_len - tec_acc) * 100, 0))
    arr_tem_percentage.append(round((avg_tec_tem/tec_len - tec_tem) * 100, 0))


#     insert eda and baseline
arr_eda.insert(0, tec_eda)
arr_bvc.insert(0, tec_bvc)
arr_acc.insert(0, tec_acc)
arr_tem.insert(0, tec_tem)

arr_eda.insert(0, 'EDA')
arr_bvc.insert(0, 'BVC')
arr_acc.insert(0, 'ACC')
arr_tem.insert(0, 'TEM')


header = ['sensor', 'baseline', 'jitter', 'scale', 'magWarp', 'timeWarp', 'rotation', 'permutation']

with open('table_accuracy.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerow(arr_eda)
    writer.writerow(arr_bvc)
    writer.writerow(arr_acc)
    writer.writerow(arr_tem)
    
    writer.writerow([])


    arr_eda_percentage.insert(0, 0)
    arr_bvc_percentage.insert(0, 0)
    arr_acc_percentage.insert(0, 0)
    arr_tem_percentage.insert(0, 0)
    
    arr_eda_percentage.insert(0, 'EDA')
    arr_bvc_percentage.insert(0, 'BVC')
    arr_acc_percentage.insert(0, 'ACC')
    arr_tem_percentage.insert(0, 'TEM')

    writer.writerow(arr_eda_percentage)
    writer.writerow(arr_bvc_percentage)
    writer.writerow(arr_acc_percentage)
    writer.writerow(arr_tem_percentage)


# In[ ]:





# In[ ]:




