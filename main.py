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

    # A DataFrame as an array
    numeric_feature_names = ['timestamp', 'X', 'Y', 'Z', 'Sleep', 'Quality']
    numeric_features = table[numeric_feature_names]
    # print(numeric_features.head())
    # tensor = tf.convert_to_tensor(numeric_features)
    # print(tensor)

    # creating DataFrame
    df = pd.DataFrame(table)
    # print(df)
    # converting timestamp
    timestamp_col = pd.to_datetime(df['timestamp'], unit='s')
    df['timestamp'] = timestamp_col
    # print(df)
    # write to csv file the date (I want to temporally visualize it)
    with open('original.csv', 'w') as csvfile:
        fieldnames = ['Date']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(0, 50):
            writer.writerow({'Date': timestamp_col[i]})

    # downsampling
    minutes = df.resample('10Min', on='timestamp')
    # print(minutes)
    with open('modified.csv', 'w') as csvfile:
        fieldnames = ['Date']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(0, 50):
            writer.writerow({'Date': df['timestamp'][i]})



if __name__ == "__main__":
    try:
        read_data_locally()
        # read_data_OneDrive()
    except Exception as message:
        print(f"MY FAULT")
