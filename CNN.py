# import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
# https://blog.finxter.com/building-a-one-dimensional-convolutional-network-in-python-using-tensorflow/
# https://www.geeksforgeeks.org/pandas/python-read-csv-using-pandas-read_csv/

df = pd.read_csv("PData.CSV",sep=',',header=None)
print(df)

# model = tf.keras.Sequential([
#     tf.keras.layers.Conv1D(64, 3, activation='relu', input_shape=(100, 1)),
#     tf.keras.layers.MaxPooling1D(2),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(60, activation='relu'),
#     tf.keras.layers.Dense(2, activation='softmax')
# ]).compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

