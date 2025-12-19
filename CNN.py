import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from datetime import datetime
import pickle
import sys
import random
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, LSTM, Dense
from sklearn.model_selection import StratifiedKFold
# Reference resource links (your comments preserved)
# 30fps max length of a video was 7 so 210 
# 68 features per frame

MAX_F = 60          # Max number of frames used per sample
FEATURES = 66       # Number of pose features per frame

# ChatGPT assisted with the padding function
def pad(data):
    # Pads or truncates each video's frame sequence to MAX_F length
    if len(data) > MAX_F:
        return data[:MAX_F]

    elif len(data) < MAX_F:
        pad = []
        i = 0
        # Create zero-padding frames
        while i < MAX_F - len(data):
            temp = []
            x = 0
            while x < FEATURES:
                temp.append(0)
                x += 1
            pad.append(temp)
            i += 1
        pad = np.array(pad)
        
        # ___________________________________________
        return np.vstack((data, pad))
        # ____________________________________________

    else:
        return data




def transform(video_data, c):

    if c == 0:
        # original
        return pad(video_data)

    video = video_data.copy()

    if c == 1:
        # Gaussian noise
        video += np.random.normal(0, 0.05, video.shape)

    elif c == 2:
        # Random per-feature scaling
        scale = np.random.uniform(0.9, 1.1, video.shape)
        video *= scale

    elif c == 3:
        # Noise + scaling
        video += np.random.normal(0, 0.03, video.shape)
        video *= np.random.uniform(0.95, 1.05, video.shape)

    # Ensure padded final shape
    video = pad(video)
    return video


if len(sys.argv) != 1:

    df = pd.read_csv("PData.CSV", sep=',', header=None)   # Load CSV with no header

    labels = []     # Store class labels
    data = []       # Store raw sequences before padding

    # ChatGPT helped with the group function
    # _______________________________________
    videos = df.groupby(1)      # Group by video ID

    # Loop over videos, building (label, data) lists
    for video_id, video in videos:
        label = video.iloc[0, 0]           # First column = class label
        video_data = video.iloc[:, 2:].values 
        labels.append(label)
        data.append(video_data)
    
# Pose sequence
    # ________________________________________

    # Encode labels (string → numeric)
    labels = LabelEncoder().fit_transform(labels)

    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        data,
        labels,
        test_size=float(sys.argv[3]),
        random_state=42,
        stratify=labels
    )

    print("Train samples:", len(X_train))
    print("Test samples:", len(X_test))

    augdata = []     # augmented training videos
    auglabels = []   # corresponding labels

    # Perform augmentations for each training video
    for video, label in zip(X_train, y_train):
        for i in range(3):   # three versions: orig, noise, scale
            aug = transform(video, i)
            augdata.append(aug)
            auglabels.append(label)

    # Pad test set
    X_test = np.array([pad(x) for x in X_test])

    # Compute class weights for imbalanced datasets
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(auglabels),
        y=auglabels
    )
    class_weight_dict = dict(enumerate(class_weights))

    # One-hot encoding
    auglabels = to_categorical(auglabels, num_classes=4)
    y_test = to_categorical(y_test, num_classes=4)

    augdata = np.array(augdata)

    # Display class distribution
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"Class {u}: {c} samples ({100*c/len(labels):.2f}%)")

    # Model definition
    model = Sequential([
        Input(shape=(MAX_F, FEATURES)),     # input layer
        Conv1D(64, 3, activation='relu'),
        Conv1D(128, 3, activation='relu'),
        MaxPooling1D(2),
        Dropout(0.5),
        LSTM(32, return_sequences=False),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(4, activation='softmax')
    ])

    # Compile
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    # Train
    model.fit(
        x=augdata,
        y=auglabels,
        epochs=int(sys.argv[1]),
        batch_size=int(sys.argv[2]),
        verbose=2,
        validation_data=(X_test, y_test),
        shuffle=True,
        class_weight=class_weight_dict,
        callbacks=[reduce_lr, early_stop]
    )

    # Save model to disk
    model_file = f"e{sys.argv[1]}b{sys.argv[2]}vs{sys.argv[3]}.h5"
    model.save(model_file)
    print(f"Model saved to {model_file}")

else:
    print("Needs command line arguments: epoch, batch size, and validation split")
