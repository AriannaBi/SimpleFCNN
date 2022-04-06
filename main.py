import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import csv
import tensorflow as tf
import torch
from enum import Enum

def type_file(name):
    print("type_file")
    type_file = name.split('/')[2][:3]
    match type_file:
        case 'EDA':
            n = 1
            return 0, n
        case 'BVP':
            n = 2
            return '500ms', n
        case 'ACC':
            n = 3
            return '250ms', n
        case 'ST':
            n = 4
            return 0, n
    return 0, 0

def read_data_locally(list_file):
    print("read_data")
    for name_file in list_file:
        # DataFrame
        table = pd.read_parquet(name_file, engine='pyarrow')

        # creating DataFrame
        df = pd.DataFrame(table)

        # converting timestamp
        timestamp_col = pd.to_datetime(df['timestamp'], unit='s')
        df['timestamp'] = timestamp_col


        arr_type = type_file(name_file)

        # create 6 matrices with rows and cols (TENSOR)
        # i - z axis            6
        # j - y axis (rows)     df.shape[0]
        # k - x axis (cols)     2400
        tensor = torch.empty((6, df.shape[0], 2400), dtype=torch.float64)


        # RESAMPLE the tensor
        # if 0, it's already 4Hz, otherwise it's either '250ms' or '500ms'
        if arr_type[0] != 0:
            df = df.resample(arr_type[0], on='timestamp').mean()
            df = df.reset_index()



        # for i in range(6):
        #     for j in range(df.shape[0]):
        #         for k in range(2400):
                    # print(tensor[i][j][k])


        # if ACC:
        #
        #
        #     ACC_X = df['X'].tolist()
        #     ACC_Y = df['Y'].tolist()
        #     ACC_Z = df['Z'].tolist()
        #
        # else if EDA:
        #     EDA = df['EDA'].tolist()
        # else if BVP
        #     BVP = df['Z'].tolist()
        # else if TEMP
        #     TEMP = df['Z'].tolist()


        with open('modified.csv', 'w') as csvfile:
            fieldnames = ['Date', 'X', 'Y', 'Z']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(0, df.shape[0]):
                writer.writerow({'Date': df['timestamp'][i], 'X': df['X'][i] , 'Y': df['Y'][i], 'Z':df['Z'][i]})

    

if __name__ == "__main__":
    try:
        list_files = ['./files/ACC_22.parquet']
        read_data_locally(list_files)
    except Exception as message:
        print(f"MY FAULT")
