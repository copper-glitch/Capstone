# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from datetime import datetime
import pickle
import sys
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
# https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
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
def transform(video_data,c):
    if c == 0:
        video_data = video_data * .90
    elif c == 1:
        video_data = video_data * 1.10
    elif c == 2:
        video_data = video_data * 1.15
    elif c == 3:
        video_data = video_data * 0.95
    elif c == 4:
        video_data = video_data * 1.05
    elif c == 5:
        video_data+=20
    elif c == 6:
        video_data-=30
    elif c == 7:
        video_data+=40
    elif c == 8:
        video_data-=50
    elif c == 9:
        video_data+=50
    return video_data
if len(sys.argv)!=1:
    df = pd.read_csv("PData copy.CSV",sep=',',header=None)

    labels = []
    data = []

    # chat gpt helpd with the group function
    # _______________________________________
    videos=df.groupby(1)
    for video_id, video in videos:
        label = video.iloc[0,0]
        video_data = video.iloc[:,2:].values
        if label == 2:
            for i in range(10):
                video_copy = video_data.copy() 
                transform(video_copy,i)
                video_data=pad(video_data)
                labels.append(label)
                data.append(video_data)                  
        else:
            video_data=pad(video_data)
            labels.append(label)
            data.append(video_data)
    #________________________________________
    data=np.array(data)
    labels = LabelEncoder().fit_transform(labels)
    class_weights = class_weight.compute_class_weight(class_weight='balanced',classes=np.unique(labels),y=labels)
    class_weight_dict = dict(enumerate(class_weights))
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"Class {u}: {c} samples ({100*c/len(labels):.2f}%)")
    model = tf.keras.Sequential([
    tf.keras.layers.Masking(0.0, input_shape=(MAX_F, FEATURES)),
    tf.keras.layers.Conv1D(64, 5, activation='relu'),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.LSTM(32, return_sequences=False),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])
    model.compile(
        optimizer='adam', 
        loss='categorical_crossentropy',          
        metrics=['accuracy'])
    labels = to_categorical(LabelEncoder().fit_transform(labels), num_classes=2)
    X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=float(sys.argv[3]), random_state=42,stratify=labels)
    
    model.fit(x=X_train,y=y_train,epochs=int(sys.argv[1]),batch_size=int(sys.argv[2]),verbose=2,validation_data=(X_test,y_test),shuffle=True,class_weight=class_weight_dict)
    model_pkl_file = f"e{sys.argv[1]}b{sys.argv[2]}vs{sys.argv[3]}.pkl"  
    
    with open(model_pkl_file, 'wb') as file:  
        pickle.dump(model, file)
else:
    print("Needs command line arguments: epoch, batch size, and validation split")
