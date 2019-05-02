import os
import sys
import glob
import argparse

from keras import __version__
from keras.applications.inception_v3 import InceptionV3, preprocess_input

from keras.preprocessing.image import ImageDataGenerator
import wandb
from wandb.keras import WandbCallback
from keras.layers import Dense, Flatten, GlobalAveragePooling2D 
from keras.models import Sequential, Model
from keras.callbacks import Callback
from keras.optimizers import SGD
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import keras
import subprocess
import os
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten


import wandb
from wandb.keras import WandbCallback

run = wandb.init()
config = run.config
config.first_layer_convs = 32
config.first_layer_conv_width = 5
config.first_layer_conv_height = 5
config.dropout = 0.2
config.dense_layer_size = 128
config.img_width = 299
config.img_height = 299
config.batch_size = 32
config.num_epochs = 20

input_shape = (48, 48, 1)


def add_new_last_layer(base_model, nb_classes):
    """Add last layer to the convnet
    Args:
        base_model: keras model excluding top
        nb_classes: # of classes
    Returns:
        new keras model with last layer
      """
    x = base_model.output
    x = Dense(config.fc_size, activation='relu')(x) #new FC layer, random init
    predictions = Dense(nb_classes, activation='softmax')(x) #new softmax layer
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def load_fer2013():
    if not os.path.exists("fer2013"):
        print("Downloading the face emotion dataset...")
        subprocess.check_output("curl -SL https://www.dropbox.com/s/opuvvdv3uligypx/fer2013.tar | tar xz", shell=True)
    data = pd.read_csv("fer2013/fer2013.csv")
    pixels = data['pixels'].tolist()
    width, height = 48, 48
    faces = []
    for pixel_sequence in pixels:
        face = np.asarray(pixel_sequence.split(' '), dtype=np.uint8).reshape(width, height)
        face = cv2.resize(face.astype('uint8'), (width, height))
        faces.append(face.astype('float32'))

    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)
    emotions = pd.get_dummies(data['emotion']).as_matrix()

    val_faces = faces[int(len(faces) * 0.8):]
    val_emotions = emotions[int(len(faces) * 0.8):]
    train_faces = faces[:int(len(faces) * 0.8)]
    train_emotions = emotions[:int(len(faces) * 0.8)]
    
    return train_faces, train_emotions, val_faces, val_emotions


# loading dataset

train_faces, train_emotions, val_faces, val_emotions = load_fer2013()
num_samples, num_classes = train_emotions.shape

train_faces /= 255.
val_faces /= 255.

model = Sequential()
model.add(Conv2D(32,
                 (config.first_layer_conv_width, config.first_layer_conv_height),
                 input_shape=(48, 48, 1),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(4,4)))
model.add(Dropout(0.5))
model.add(Conv2D(64,(5,5),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(4,4)))
model.add(Dropout(0.5))
model.add(Flatten(input_shape=input_shape))
model.add(Dense(config.dense_layer_size, activation='relu'))
model.add(Dense(config.dense_layer_size, activation='relu'))

model.add(Dense(num_classes, activation="softmax"))

model.compile(optimizer='adam', loss='categorical_crossentropy',
metrics=['accuracy'])
config.total_params = model.count_params()
model.fit(train_faces, train_emotions, batch_size=config.batch_size,
        epochs=config.num_epochs, verbose=1, callbacks=[
            WandbCallback(data_type="image", labels=["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"])
        ], validation_data=(val_faces, val_emotions))


model.save("emotion.h5")



