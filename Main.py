#!/usr/bin/env python
# coding: utf-8

# In[466]:


import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import csv
import glob
from itertools import chain
from keras import models
from keras import layers
import os
import keras 
import sys

from keras.models import Sequential
from keras.layers import Dense


from collections import Counter
import matplotlib.pyplot as plt
from keras.callbacks import History

np.set_printoptions(threshold=sys.maxsize)


# In[467]:


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
#     print(name_file)
    if name_file == 'EDA':
        return 0, 0
    elif name_file == 'BVP':
        return '250ms', 1
    elif name_file == 'ACC':
        return '250ms', 2
    elif name_file == 'TEM':
        return 0, 5
    return -1


# In[468]:


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


# In[469]:


def append_data_and_fill_missing_zeros(ARRAY, df, value):
    for i in range(0, df.shape[0]):
        ARRAY.append(df[value][i])
#         print("BEFORE: ", len(ARRAY))

    if (len(ARRAY) % 2400) != 0:
        rounding = int(len(ARRAY)//2400 + (len(ARRAY) % 2400 > 0))
        number_cells = rounding * 2400
        missing_zeros = [0] * (number_cells - len(ARRAY))
        ARRAY = ARRAY + missing_zeros
    return ARRAY
    
    


# In[494]:


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
        missing_zeros = [0]*(int(len(EDA)/2400)-len(LABEL))
        LABEL = LABEL + missing_zeros
        
#     print(len(LABEL))
    return LABEL
        


# In[495]:


def add_zeros_to_uniform_size_array(ARRAY, missing_zeros):
    ARRAY = [0]* missing_zeros
    return ARRAY


# In[508]:


# Call function create_train_set_train_label so that we have the two zero tensors.
# for each file read the data, downsampling the data and fill the tensors.

def read_data(files):
    EDA, BVP, ACC_X, ACC_Y, ACC_Z, TEM, LABEL = [],[],[],[],[],[],[]

    
    # READ ALL THE FILES and DOWNSAMPLING
    for name_file in files:
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
        
        
        print(df.shape)
       # CREATES THE ARRAYS
        # create linear arrays as sequences of elements. for each file add severl zeros to complete a window of 2400     
        if arr_type[1] == 0:
            EDA = append_data_and_fill_missing_zeros(EDA, df, 'value')
#             print("EDA",len(EDA))
#             df['Sleep']
            LABEL = find_sleep_label(LABEL, EDA, df)
#             print("label",len(LABEL))
                
        elif arr_type[1] == 1:
            BVP = append_data_and_fill_missing_zeros(BVP, df, 'value')

        elif arr_type[1] == 2:
            ACC_X = append_data_and_fill_missing_zeros(ACC_X, df, 'X')
            ACC_Y = append_data_and_fill_missing_zeros(ACC_Y, df, 'Y')
            ACC_Z = append_data_and_fill_missing_zeros(ACC_Z, df, 'Z')
            
        elif arr_type[1] == 5:
            TEM = append_data_and_fill_missing_zeros(TEM, df, 'value')
        
        
#     print("LEN LABEL",len(LABEL))
#     if each file has the same length it doesn make sense to check it. only in between files fill the row of zeros
#     and then start from the next row a new file
#     if the files are not the same quantity, it's better to throw error?
    uniform_len = max(len(EDA),len(BVP),len(ACC_X),len(ACC_Y),len(ACC_Z),len(TEM))    
#     print(len(EDA), ' ',len(BVP), ' ',len(ACC_X), ' ',len(ACC_Y), ' ',len(ACC_Z), ' ',len(TEM) )
#     check train_set
    if len(EDA) != uniform_len:
        EDA = EDA + [0]* (uniform_len - len(EDA))
    if len(BVP) != uniform_len:
        BVP = BVP + [0]* (uniform_len - len(BVP))
    if len(ACC_X) != uniform_len:
        ACC_X = ACC_X + [0]* (uniform_len - len(ACC_X))
    if len(ACC_Y) != uniform_len:
        ACC_Y = ACC_Y + [0]* (uniform_len - len(ACC_Y))
    if len(ACC_Z) != uniform_len:
        ACC_Z = ACC_Z + [0]* (uniform_len - len(ACC_Z))
    if len(TEM) != uniform_len:
        TEM = TEM + [0]* (uniform_len - len(TEM))
        
#     print(len(EDA), ' ',len(BVP), ' ',len(ACC_X), ' ',len(ACC_Y), ' ',len(ACC_Z), ' ',len(TEM) )


#     print(len(EDA_LABEL), ' ',len(BVP_LABEL), ' ',len(ACC_X_LABEL), ' ',len(ACC_Y_LABEL), ' ',len(ACC_Z_LABEL), ' ',len(TEM_LABEL) )
#     print(len(EDA_LABEL), ' ',len(BVP_LABEL), ' ',len(ACC_X_LABEL), ' ',len(ACC_Y_LABEL), ' ',len(ACC_Z_LABEL), ' ',len(TEM_LABEL) )
    # convert the list to a numpy array
    train_set = np.array([EDA, BVP, ACC_X, ACC_Y, ACC_Z, TEM],dtype=np.float64)
    train_label = np.array([LABEL], dtype=np.float64)
    

    # Reshape the tensor train set, counting the number of rows dividing by 2400
    row_training_set = uniform_len // 2400
    if (uniform_len % 2400) != 0:
        return "Error in missing_zeros"
    train_set = train_set.reshape(6,row_training_set,2400)
#     Reshape the tensor train label
#     train_label = train_label.reshape(6,uniform_len_label, 1)
    

    train_set = np.dstack(train_set)
    train_label = train_label.reshape(train_label.shape[1], 1)
    print(train_set.shape)
    print(train_label.shape)
    
    
    model = Sequential()
    model.add(Dense(16, activation='relu', input_shape=(2400, 6), name="layer1"))
    model.add(Dense(12, activation='relu', name="layer2"))
    model.add(Dense(1, activation='sigmoid', name="layer3"))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(np.array(train_set), np.array(train_label), epochs=15, batch_size=128)

    print("END")
    
    
    
# Retrieve the files and crete the list
list_file = []
# for i in range(1,2):
#     list_file.append(sorted(glob.glob("../trial/S0"+ str(i) +"/*.parquet")))

list_file2 = np.array(list_file)
list_file = list(list_file2.flat)

list_file = ["../trial/S01/EDA_01.parquet","../trial/S01/BVP_01.parquet",
            "../trial/S01/EDA_02.parquet","../trial/S01/BVP_02.parquet"]


print(list_file)

read_data(list_file)




# In[502]:


train_data_labels = np.array([[0], [1], [1]])
train_data_labels.shape


# In[ ]:




