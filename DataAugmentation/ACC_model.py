import numpy as np
import tensorflow as tf
import random as rn
import os
import sys
from tensorflow.keras.optimizers import SGD
import pandas as pd
import pyarrow.parquet as pq
import csv
from keras import models
from keras import layers
from keras import Input
import keras
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from keras.layers import Dense
from keras.callbacks import History
from tensorflow.keras.metrics import Recall
from tensorflow.keras.metrics import Precision
from scipy.interpolate import CubicSpline      # for warping
from transforms3d.axangles import axangle2mat  # for rotation
from matplotlib import pyplot

np.set_printoptions(threshold=sys.maxsize)
# import augment_data
from augment_data import Augmented_data


best_lr = 0.001
epoch = 100
batch_s = 128


def model_network(learning_rate):
    seed_value = 34567892
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    tf.random.set_seed(seed_value)

    model = Sequential()
    model.add(Input(shape=(2400,)))
    model.add(layers.BatchNormalization())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
        # opt = SGD(learning_rate=0.0001,momentum=0.4)
    opt = Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy', Recall(), Precision()])
    return model


# for each technique run tec_len times the model to obtain an average of accuracy and fill the csv table with results
techniques = ['jitter', 'scale', 'magWarp', 'timeWarp', 'permutation']

line_csv_acc, line_csv_recall, line_csv_precision = [], [], []
std_accuracy, std_recall, std_precision = [], [], []
arr_eda_percentage = []

train_set = np.load('datasets/train_set_original.npy', encoding='ASCII')
train_label = np.load('datasets/train_label_original.npy', encoding='ASCII')
test_set = np.load('datasets/test_set.npy', encoding='ASCII')
test_label = np.load('datasets/test_label.npy', encoding='ASCII')
train_label = train_label.reshape(train_label.shape[0], 1)
test_label = test_label.reshape(test_label.shape[0], 1)

train_set_arr = train_set[2]
test_set_arr = test_set[2]

model = model_network(best_lr)

# ORIGINAL SET
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
history = model.fit(train_set_arr, train_label, epochs=epoch, batch_size=batch_s, shuffle=True, verbose=0,
                    callbacks=[callback])
scores1 = model.evaluate(test_set_arr, test_label, verbose=0)
print("evaluate original: ", scores1)
original_accuracy = scores1[1]
line_csv_acc.append(round(scores1[1] * 100, 4))
line_csv_recall.append(round(scores1[2] * 100, 4))
line_csv_precision.append(round(scores1[3] * 100, 4))
std_accuracy.append(0)
std_recall.append(0)
std_precision.append(0)

# --------------------------------------------------------------------

for technique in techniques:
    print(technique)

    recall_original = 0
    accuracies, recalls, precisions = [], [], []

    tec_len = 10

    # loop tec_len times to get the average of a tecnique
    for avg_t in range(0, tec_len):
        train_set = np.load('datasets/train_set_original.npy', encoding='ASCII')
        train_label = np.load('datasets/train_label_original.npy', encoding='ASCII')
        test_set = np.load('datasets/test_set.npy', encoding='ASCII')
        test_label = np.load('datasets/test_label.npy', encoding='ASCII')
        train_label = train_label.reshape(train_label.shape[0], 1)
        test_label = test_label.reshape(test_label.shape[0], 1)

        train_set_arr_augment, label_set_augmented = Augmented_data.augment_data(train_set[2], train_label, technique)

        test_set_arr = test_set[2]
        model = model_network(best_lr)

        # AUGMENTATION
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
        history = model.fit(train_set_arr_augment, label_set_augmented, epochs=epoch, batch_size=batch_s,
                            shuffle=True, verbose=0, callbacks=[callback])
        scores2 = model.evaluate(test_set_arr, test_label, verbose=0)
        #     --------------------------------------------------------------------
        print("evaluate augmented : ", technique, scores2)

        accuracies.append(scores2[1])
        recalls.append(scores2[2])
        precisions.append(scores2[3])

    line_csv_acc.append(round(np.mean(accuracies) * 100, 4))
    line_csv_recall.append(round(np.mean(recalls) * 100, 4))
    line_csv_precision.append(round(np.mean(precisions) * 100, 4))
    std_accuracy.append(round(np.std(accuracies) * 100, 4))
    std_recall.append(round(np.std(recalls) * 100, 4))
    std_precision.append(round(np.std(precisions) * 100, 4))
    print(accuracies)
    print(recalls)
    print(precisions)
    print("mean: {} -- std: (+/- {}".format(np.mean(accuracies), np.std(accuracies)))

    arr_eda_percentage.append(round((np.mean(accuracies) - original_accuracy) * 100, 4))

line_csv_acc.insert(0, 'ACC')
std_accuracy.insert(0, 'STD ACCURACY')
line_csv_recall.insert(0, 'RECALL')
std_recall.insert(0, 'STD RECALL')
line_csv_precision.insert(0, 'PRECISION')
std_precision.insert(0, 'STD PRECISION')

header = ['sensor', 'baseline', 'jitter', 'scale', 'magWarp', 'timeWarp', 'permutation']

with open('table_accuracy_ACC.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerow(line_csv_acc)
    writer.writerow(std_accuracy)
    writer.writerow(line_csv_recall)
    writer.writerow(std_recall)
    writer.writerow(line_csv_precision)
    writer.writerow(std_precision)

    writer.writerow([])
    arr_eda_percentage.insert(0, 0)
    arr_eda_percentage.insert(0, 'ACC')
    writer.writerow(arr_eda_percentage)

