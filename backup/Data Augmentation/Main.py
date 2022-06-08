#!/usr/bin/env python
# coding: utf-8

# find . -name ".DS_Store" -delete

import numpy as np
import tensorflow as tf
import random as rn
import os
import sys
os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
# random seed for numpy
np.random.seed(37)
# random seed for python
rn.seed(1254)
# random seed for tensorflow
tf.random.set_seed(89)

import pandas as pd
import pyarrow.parquet as pq

import csv
import glob
# from itertools import chain
from keras import models
from keras import layers
import keras 

from sklearn.model_selection import train_test_split

from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense

from collections import Counter
import matplotlib.pyplot as plt
from keras.callbacks import History

np.set_printoptions(threshold=sys.maxsize)


# given a path like "../trial/S01/ACC_01.parquet" return:
# the number of ms that a file needs in order to do the downsampling;
# and return the type among:
# - 0 (EDA)
# - 1 (BVP)
# - 2 (ACC)
# - 3 (TEM) 

# If the file is incorrect return -1
def type_file(name):
    name_file = name.split('/')[3][:3]
    if name_file == 'EDA':
        return 0, 0
    elif name_file == 'BVP':
        return '250ms', 1
    elif name_file == 'ACC':
        return '250ms', 2
    elif name_file == 'TEM':
        return 0, 5
    return -1

# Function to determine the most frequent element in a list:
# [0,0,0,1,1] return 0
# [0,0,1,1,1] return 1

def most_frequent(List):
    counter = 0
    num = List[0]

    for i in List:
        curr_frequency = List.count(i)
        if(curr_frequency> counter):
            counter = curr_frequency
            num = i
    return num

# listt = [0,1,0,1]
# print(most_frequent(listt))


def append_data_and_fill_missing_nan(ARRAY, df, value):
    for i in range(0, df.shape[0]):
        ARRAY.append(df[value][i])

    if (len(ARRAY) % 2400) != 0:
        rounding = int(len(ARRAY)//2400 + (len(ARRAY) % 2400 > 0))
        number_cells = rounding * 2400
        missing_zeros = [np.nan] * (number_cells - len(ARRAY))
        ARRAY = ARRAY + missing_zeros
    return ARRAY


def find_sleep_label(LABEL, EDA, df):
    i = 0
    list_sleep = []
#     create the list_sleep with all the values of the colums 'Sleep'
    for j in range(0, df.shape[0]):
        value = df['Sleep'][j].astype(np.float64)
        list_sleep.append(value)

#   for each 2400 window choose if 0 or 1, and append it to ARRAR_LABEL
    while i in range(0, len(list_sleep)- 2399):
        list_sleep_window = list_sleep[i:(i+2400)]
        label_sleep = most_frequent(list_sleep_window)
        LABEL.append(label_sleep)
        i += 2400
        
#     if EDA len is greather than ARRAY_LABEL, uniform it to the same length
    if (len(LABEL) < int(len(EDA)/2400)):
        missing_zeros = [np.nan]*(int(len(EDA)/2400)-len(LABEL))
        LABEL = LABEL + missing_zeros
        
#     print(len(LABEL))
    return LABEL
        

def add_zeros_to_uniform_size_array(ARRAY, missing_zeros):
    ARRAY = [np.nan]* missing_zeros
    return ARRAY


def remove_nan_arrays(ARRAY, ARR2, ARR3, ARR4, ARR5, ARR6, LABEL):
    array_zeros = np.zeros(2400)
    
    array_boolean = np.isnan(ARRAY).any(axis=1)

    # iterate over each row.
    # if delete a "true" then continue in a while untill it goes 1 step forward
    i = 0
    while i < len(array_boolean):
        while array_boolean[i]:
            ARRAY = np.delete(ARRAY, i, 0)
            ARR2 = np.delete(ARR2, i, 0)
            ARR3 = np.delete(ARR3, i, 0)
            ARR4 = np.delete(ARR4, i, 0)
            ARR5 = np.delete(ARR5, i, 0)
            ARR6 = np.delete(ARR6, i, 0)
            LABEL = np.delete(LABEL, i, 0)
            array_boolean = np.delete(array_boolean, i)
        i += 1
    
#     print(ARRAY.shape)
    ARRAY = ARRAY[:-1]
    ARR2 = ARR2[:-1]
    ARR3 = ARR3[:-1]
    ARR4 = ARR4[:-1]
    ARR5 = ARR5[:-1]
    ARR6 = ARR6[:-1]
    LABEL = LABEL[:-1]
    return ARRAY, ARR2, ARR3, ARR4, ARR5, ARR6, LABEL


# from array 1D, i create an array 2D with the last row as 0.
def reshape_arrays(EDA):
    EDA = np.array(EDA)
#     print("eda before: ",EDA.shape)
    EDA = EDA.reshape(1,len(EDA)//2400, 2400)
    array_zeros = np.zeros(2400)
    EDA = EDA[0]
    EDA = np.concatenate((EDA, [array_zeros]), axis=0)
    
#     print("eda after: ",EDA.shape)
    return EDA


# from array 1D, i append a 0 in the last row.
def reshape_label(LABEL):
    LABEL = np.array(LABEL)
#     print("LABEL before: ", LABEL.shape)
#     LABEL = LABEL.reshape(1,len(LABEL), 1)
#     array_zeros = np.zeros(2400)
#     LABEL = LABEL[0]
    LABEL = np.concatenate((LABEL, [0]), axis=0)
    
#     print("LABEL after: ",LABEL.shape)
    return LABEL


# Call function create_train_set_train_label so that we have the two zero tensors.
# for each file read the data, downsampling the data and fill the tensors.

def read_data(files):
    EDA, BVP, ACC_X, ACC_Y, ACC_Z, TEM, LABEL = [],[],[],[],[],[],[]


    # READ ALL THE FILES and DOWNSAMPLING
    for name_file in files:
#         print(name_file)
        rounding, number_cells, missing_zeros = 0, 0, []
        # DataFrame
        table = pd.read_parquet(name_file, engine='pyarrow')
        # creating DataFrame
        df = pd.DataFrame(table)
        # converting timestamp
        timestamp_col = pd.to_datetime(df['timestamp'], unit='s')
        df['timestamp'] = timestamp_col
        # get if file it's among EDA, BVP, ACC, ST, otherwise return error
        arr_type = type_file(name_file)
        if arr_type == -1:
            print("Error in the file name")
            return -1

        # RESAMPLE the tensor
        # 0 means it's already 4Hz, otherwise resample with '250ms'
        if arr_type[0] != 0:
            df = df.resample(arr_type[0], on='timestamp').mean()
            df = df.reset_index()
        
        # remove the          
        
#         print(df.shape)
       # CREATES THE ARRAYS
        # create linear arrays as sequences of elements. for each file add several zeros to complete a window of 2400     
        if arr_type[1] == 0:
            EDA = append_data_and_fill_missing_nan(EDA, df, 'value')
            LABEL = find_sleep_label(LABEL, EDA, df)
                
        elif arr_type[1] == 1:
            BVP = append_data_and_fill_missing_nan(BVP, df, 'value')

        elif arr_type[1] == 2:
            ACC_X = append_data_and_fill_missing_nan(ACC_X, df, 'X')
            ACC_Y = append_data_and_fill_missing_nan(ACC_Y, df, 'Y')
            ACC_Z = append_data_and_fill_missing_nan(ACC_Z, df, 'Z')
            
        elif arr_type[1] == 5:
            TEM = append_data_and_fill_missing_nan(TEM, df, 'value')
        
        
#     print("LEN LABEL",len(LABEL))
#     if each file has the same length it doesn make sense to check it. only in between files fill the row of zeros
#     and then start from the next row a new file
#     if the files are not the same quantity, it's better to throw error?
    uniform_len = max(len(EDA),len(BVP),len(ACC_X),len(ACC_Y),len(ACC_Z),len(TEM))    
#     print(len(EDA), ' ',len(BVP), ' ',len(ACC_X), ' ',len(ACC_Y), ' ',len(ACC_Z), ' ',len(TEM) )
#     check train_set
    if len(EDA) != uniform_len:
        EDA = EDA + [np.nan]* (uniform_len - len(EDA))
    if len(BVP) != uniform_len:
        BVP = BVP + [np.nan]* (uniform_len - len(BVP))
    if len(ACC_X) != uniform_len:
        ACC_X = ACC_X + [np.nan]* (uniform_len - len(ACC_X))
    if len(ACC_Y) != uniform_len:
        ACC_Y = ACC_Y + [np.nan]* (uniform_len - len(ACC_Y))
    if len(ACC_Z) != uniform_len:
        ACC_Z = ACC_Z + [np.nan]* (uniform_len - len(ACC_Z))
    if len(TEM) != uniform_len:
        TEM = TEM + [np.nan]* (uniform_len - len(TEM))
        
#     print(len(EDA_LABEL), ' ',len(BVP_LABEL), ' ',len(ACC_X_LABEL), ' ',len(ACC_Y_LABEL), ' ',len(ACC_Z_LABEL), ' ',len(TEM_LABEL) )
    
    
    # convert the list to a numpy array
#     removing rows with nan
    EDA_np, BVP_np, TEM_np = np.array(EDA),np.array(BVP),np.array(TEM)
    ACC_X_np, ACC_Y_np, ACC_Z_np = np.array(ACC_X),np.array(ACC_Y),np.array(ACC_Z)
    LABEL_np = np.array(LABEL)

    
    EDA_np = reshape_arrays(EDA)
    BVP_np = reshape_arrays(BVP)
    ACC_X_np = reshape_arrays(ACC_X)
    ACC_Y_np = reshape_arrays(ACC_Y)
    ACC_Z_np = reshape_arrays(ACC_Z)
    TEM_np = reshape_arrays(TEM)
    LABEL_np =  reshape_label(LABEL)

    

    EDA_np, BVP_np, ACC_X_np, ACC_Y_np, ACC_Z_np, TEM_np,LABEL_np = remove_nan_arrays(EDA_np, BVP_np, ACC_X_np, ACC_Y_np, ACC_Z_np, TEM_np, LABEL_np)
# each file has different nan rows
    BVP_np, EDA_np, ACC_X_np, ACC_Y_np, ACC_Z_np, TEM_np,LABEL_np = remove_nan_arrays(BVP_np, EDA_np, ACC_X_np, ACC_Y_np, ACC_Z_np, TEM_np,LABEL_np)
    ACC_X_np, EDA_np, BVP_np, ACC_Y_np, ACC_Z_np, TEM_np,LABEL_np = remove_nan_arrays(ACC_X_np, EDA_np, BVP_np, ACC_Y_np, ACC_Z_np, TEM_np,LABEL_np)
    ACC_Y_np, EDA_np, BVP_np, ACC_X_np, ACC_Z_np, TEM_np,LABEL_np = remove_nan_arrays(ACC_Y_np, EDA_np, BVP_np, ACC_X_np, ACC_Y_np, TEM_np,LABEL_np)
    ACC_Z_np, EDA_np, BVP_np, ACC_X_np, ACC_Y_np, TEM_np,LABEL_np = remove_nan_arrays(ACC_Z_np, EDA_np, BVP_np, ACC_X_np, ACC_Y_np, TEM_np,LABEL_np)
    TEM_np, EDA_np, BVP_np, ACC_X_np, ACC_Y_np, ACC_Z_np,LABEL_np = remove_nan_arrays(TEM_np, EDA_np, BVP_np, ACC_X_np, ACC_Y_np, ACC_Z_np,LABEL_np)

    
    print(EDA_np.shape)
    print(BVP_np.shape)
    print(ACC_X_np.shape)
    print(ACC_Y_np.shape)
    print(ACC_Z_np.shape)
    print(TEM_np.shape)
    print(LABEL_np.shape)
    

    EDA = EDA_np.flatten().tolist()
    BVP = BVP_np.flatten().tolist()
    ACC_X = ACC_X_np.flatten().tolist()
    ACC_Y = ACC_Y_np.flatten().tolist()
    ACC_Z = ACC_Z_np.flatten().tolist()
    TEM = TEM_np.flatten().tolist()
#     print(len(EDA))
    
    
    
    train_set = np.array([EDA, BVP, ACC_X, ACC_Y, ACC_Z, TEM],dtype=np.float64)
    train_label = np.array([LABEL_np], dtype=np.float64)
    

    # Reshape the tensor train set, counting the number of rows dividing by 2400
    uniform_len = max(len(EDA),len(BVP),len(ACC_X),len(ACC_Y),len(ACC_Z),len(TEM))    
    row_training_set = uniform_len // 2400
    if (uniform_len % 2400) != 0:
        return "Error in missing_zeros"
    train_set = train_set.reshape(6,row_training_set,2400)

    print(train_set.shape)
    train_set = np.dstack(train_set)
    print(train_label.shape)
    train_label = train_label.reshape(train_label.shape[1], 1)
    
    print(train_label.shape)
    
    
    
    print("END read data")
    return train_set, train_label


def create_datasets(start_session, end_session):
#     create list of folders
    listt = os.listdir("../Sessions/")

#   how many users? for now only 1
    list_dir = sorted(listt)[0:1]
#     list_dir = sorted(listt)
    list_path_dir = []
    for elem in list_dir:
        list_path_dir.append("../Sessions/" + elem)

#     read files in the folder and create a list of files 
    list_ordered_files = []
    
    for folder in list_path_dir:
#         print(folder)
        list_files = []
        for infile in os.listdir(folder):
            list_files.append(infile)
        list_files = sorted(list_files)
#         print(list_files)

    #     divide the list into 4 sections: acc, bvp, eda, temp
    #     take one from each and construct a list 
    #     the order is really important while reading files
        
        acc = []
        bvp = []
        eda = []
        tem = []
        for elem in (list_files):
            if elem[:3] == "ACC":
                acc.append(elem)
            elif elem[:3] == "BVP":
                bvp.append(elem)
            elif elem[:3] == "EDA":
                eda.append(elem)
            elif elem[:3] == "TEM":
                tem.append(elem)

        acc = acc[start_session:end_session]
        bvp = bvp[start_session:end_session]
        eda = eda[start_session:end_session]
        tem = tem[start_session:end_session]
    #     create a list with that order: [acc1, bvp1, eda1, tem1, acc2, bvp2, eda2, tem2,...]
        for i in range(len(acc)):
            list_ordered_files.append(folder + "/" + acc[i])
            list_ordered_files.append(folder + "/" + bvp[i])
            list_ordered_files.append(folder + "/" + eda[i])
            list_ordered_files.append(folder + "/" + tem[i])
#     print(list_ordered_files)
        
    
    return read_data(list_ordered_files)



# 4 is the number of sessions per user
# train_set, train_label = create_datasets(0,4)
# from 0 session to 5 session per user
# train_set, train_label =  create_datasets(0,5)


# test_set, test_label = create_datasets(5,6)


def network_model(train_set, train_label) :
    
    opt = Adam(learning_rate=0.01,beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    name="Adam")
    
    
    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=(2400, 6), name="layer1"))
    model.add(Dense(12, activation='relu', name="layer2"))
    model.add(Dense(1, activation='sigmoid', name="layer3"))

    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(np.array(train_set), np.array(train_label), epochs=10, batch_size=128)
    print("END network model")
    



network_model(train_set,train_label)

