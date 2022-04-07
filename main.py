import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import csv
import torch

def type_file(name):
    match name.split('/')[2][:3]:
        case 'EDA':
            return 0, 0
        case 'BVP':
            return '250ms', 1
        case 'ACC':
            return '250ms', 2
        case 'TEM':
            return 0, 5
    return -1

def read_data_locally(list_file):
    for name_file in list_file:
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
            print("Error in the file")
            return -1

        # RESAMPLE the tensor
        # if 0 it's already 4Hz, otherwise it's either '250ms'
        if arr_type[0] != 0:
            print("resample stage")
            df = df.resample(arr_type[0], on='timestamp').mean()
            df = df.reset_index()

        # create 6 matrices with rows and cols ( create TENSOR)
        # i - z axis            6
        # j - y axis (rows)     df.shape[0]/2400 (how many windows)
        # k - x axis (cols)     2400
        number_rows = int(df.shape[0] // 2400)
        tensor = torch.empty((6, number_rows, 2400), dtype=torch.float64)
        print(tensor)
        # FILL the tensor
        # if not ACC, fill the #st matrix
        if arr_type[1] != 2:
            print("fill tensor stage not ACC")
            position_element_in_column = 0
            for j in range(number_rows):
                for k in range(2400):
                    tensor[arr_type[1]][j][k] = df['value'][position_element_in_column]
                    position_element_in_column +=  1
            print(tensor)

        # otherwise fill ACC_X, ACC_Y, ACC_Z
        elif arr_type[1] == 2:
            print("fill tensor stage of ACC")
            # list_axis_ACC = ['X', 'Y', 'Z']
            position_element_in_column = 0
            # for axis in list_axis_ACC:
            #     print(axis)
                # position_element_in_column = 0
                # for j in range(number_rows):
                #     for k in range(2400):
                #         tensor[arr_type[2]][j][k] = df[axis][position_element_in_column]
                #         position_element_in_column += 1
            # print(tensor)

            # position_element_in_column = 0
            # for j in range(df.shape[0]):
            #     for k in range(2400):
            #         position_element_in_column += 1
            #         tensor[arr_type[3]][j][k] = df['Y'][position_element_in_column]
            #         # print(tensor[arr_type[3]][j][k])
            #
            # position_element_in_column = 0
            # for j in range(df.shape[0]):
            #     for k in range(2400):
            #         position_element_in_column += 1
            #         tensor[arr_type[4]][j][k] = df['Z'][position_element_in_column]
            #         # print(tensor[arr_type[4]][j][k])




        # # for BVP
        # with open('modified.csv', 'w') as csvfile:
        #     fieldnames = ['Date', 'Value']
        #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        #     writer.writeheader()
        #     for i in range(0, 3000):
        #         writer.writerow({'Date': df['timestamp'][i], 'Value': df['value'][i]})

        # for ACC
        # with open('modified.csv', 'w') as csvfile:
        #     fieldnames = ['Date', 'X', 'Y', 'Z']
        #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        #     writer.writeheader()
        #     for i in range(0, df.shape[0]):
        #         writer.writerow({'Date': df['timestamp'][i], 'X': df['X'][i], 'Y': df['Y'][i], 'Z': df['Z'][i]})


if __name__ == "__main__":
    try:
        # reading a list of files
        list_files = ['./files/ACC_22.parquet']
        # list_files = ['./files/BVP_41.parquet']
        # list_files = ['./files/TEMP_25.parquet']
        # list_files = ['./files/EDA_02.parquet']
        read_data_locally(list_files)
    except Exception as message:
        print(f"MY FAULT")
