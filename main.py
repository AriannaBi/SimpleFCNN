import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import csv
import tensorflow as tf


def read_data_locally():
    # DataFrame
    table = pd.read_parquet('./ACC_22.parquet', engine='pyarrow')
    head = table.head()
    # print(table)
    # print(head)
    # types = table.dtypes
    # print(types)
    # timestamp = table.pop('timestamp')
    # print(timestamp)


    # creating DataFrame
    df = pd.DataFrame(table)

    # converting timestamp
    timestamp_col = pd.to_datetime(df['timestamp'], unit='s')
    df['timestamp'] = timestamp_col
    len = df.shape[0]

    # write to csv file the date (I want to temporally visualize it)
    # with open('original.csv', 'w') as csvfile:
    #     fieldnames = ['Date','X','Y','Z']
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #     writer.writeheader()
    #     for i in range(0, 20):
    #         writer.writerow({'Date': df['timestamp'][i], 'X': df['X'][i] , 'Y': df['Y'][i], 'Z':df['Z'][i]})


    # downsampling
    df = df.resample('250ms', on='timestamp').sum()
    # print(df.shape[0])
    df = df.reset_index()
    # ----------------------------------- write on file parquet ------------------------------------------------
    # table = df.head(20)
    # table.to_parquet('filep.parquet')
    # ----------------------------------------------------------------------------------------------------------


    with open('modified.csv', 'w') as csvfile:
        fieldnames = ['Date', 'X', 'Y', 'Z']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(0, 10000):
            writer.writerow({'Date': df['timestamp'][i], 'X': df['X'][i] , 'Y': df['Y'][i], 'Z':df['Z'][i]})

    # can't update tensors, only create
    



if __name__ == "__main__":
    try:
        read_data_locally()
    except Exception as message:
        print(f"MY FAULT")
