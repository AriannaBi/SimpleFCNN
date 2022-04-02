import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import csv


def read_data_locally():
    # DataFrame
    table = pd.read_parquet('./ACC_22.parquet', engine='pyarrow')
    head = table.head()
    # print(table)
    # print(head)
    types = table.dtypes
    # print(types)
    # timestamp = table.pop('timestamp')
    # print(timestamp)


    # creating DataFrame
    df = pd.DataFrame(table)
    # print(df)
    # converting timestamp
    timestamp_col = pd.to_datetime(df['timestamp'], unit='s')
    df['timestamp'] = timestamp_col
    # print(df)
    # write to csv file the date (I want to temporally visualize it)
    with open('original.csv', 'w') as csvfile:
        fieldnames = ['Date','X','Y','Z']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(0, 50):
            writer.writerow({'Date': df['timestamp'][i], 'X': df['X'][i] , 'Y': df['Y'][i], 'Z':df['Z'][i]})

    # downsampling
    df = df.resample('10s', on='timestamp').first()
    # print(df)
    with open('modified.csv', 'w') as csvfile:
        fieldnames = ['Date', 'X', 'Y', 'Z']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(0, 50):
            writer.writerow({'Date': df['timestamp'][i], 'X': df['X'][i] , 'Y': df['Y'][i], 'Z':df['Z'][i]})



if __name__ == "__main__":
    try:
        read_data_locally()
        # read_data_OneDrive()
    except Exception as message:
        print(f"MY FAULT")
