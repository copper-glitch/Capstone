# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from datetime import datetime
import pickle
# 30fps max length of a video was 7 so 210 
# 68 features per frame

MAX_F = 60
FEATURES = 66
# https://www.w3schools.com/python/ref_module_time.asp
# https://blog.finxter.com/building-a-one-dimensional-convolutional-network-in-python-using-tensorflow/
# https://www.geeksforgeeks.org/pandas/python-read-csv-using-pandas-read_csv/
# https://www.tensorflow.org/guide/keras/preprocessing_layers
# https://www.tensorflow.org/api_docs/python/tf/keras/utils/pad_sequences
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.iloc.html
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
# https://www.geeksforgeeks.org/deep-learning/model-fit-in-tensorflow/
# https://www.geeksforgeeks.org/deep-learning/keras-input-layer/
# https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html
# https://datascience.stackexchange.com/questions/13490/how-to-set-class-weights-for-imbalanced-classes-in-keras
# https://www.tensorflow.org/guide/keras/understanding_masking_and_padding

# Chatgpt helped me with the padding function

def pad(data):
    if len(data)>MAX_F:
        return data[:MAX_F]
    elif len(data)<MAX_F:
        pad = []
        i=0
        while i<MAX_F-len(data):
            temp = []
            x = 0
            while x<FEATURES:
                temp.append(0)
                x+=1
            pad.append(temp)
            i+=1
        pad=np.array(pad)
        
        # ___________________________________________
        return np.vstack((data,pad))
        # ____________________________________________
    else:
        return data


df = pd.read_csv("PData copy.CSV",sep=',',header=None)

labels = []
data = []

# chat gpt helpd with the group function
# _______________________________________
videos=df.groupby(1)

for video_id, video in videos:
    label = video.iloc[0,0]
    video_data = video.iloc[:,2:].values
    video_data=pad(video_data)
    labels.append(label)
    data.append(video_data)
#________________________________________
    
    
    
# print(df)
# labels = df.iloc[0,0].values  # shape: (samples, 14280)
# # print(labels)
# data = df.iloc[:,2:].values  # shape: (samples, 14280)
# print(data)
# data = np.array([pad(x.reshape(-1,FEATURES))for x in data])
# # print(data)
# data = data.reshape(-1, MAX_F, FEATURES)

data=np.array(data)

labels = LabelEncoder().fit_transform(labels)
class_weights = class_weight.compute_class_weight(class_weight='balanced',classes=np.unique(labels),y=labels)
class_weight_dict = dict(enumerate(class_weights))

model = tf.keras.Sequential([
    tf.keras.layers.Masking(0.0,input_shape=(MAX_F,FEATURES)),
    tf.keras.layers.Conv1D(32, 3, activation='relu', input_shape=(MAX_F,FEATURES)),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(.23),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x=data,y=labels,epochs=25,batch_size=21,verbose=2,validation_split=.3,shuffle=True,class_weight=class_weight_dict)

model_pkl_file = f"{datetime.now()}CNN.pkl"  
clean = ""
for x in model_pkl_file:
    if x != ":":
        clean.join(x)

with open("dave.pkl", 'wb') as file:  
    pickle.dump(model, file)
print("Written")
