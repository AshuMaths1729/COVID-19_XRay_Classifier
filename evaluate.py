"""
Author: Ashutosh Agrahari
Time: 19-Apr-2020 2:31PM GMT+5:30

Code for inferencing and testing the model.
"""

import tensorflow as tf
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.optimizers import RMSprop, Adam
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import plot_model
import matplotlib.pyplot as plt
import os


# Data
training_data_dir = "data/train" 
test_data_dir = "data/test"
valid_data_dir = "data/val"

# Hyperparams
IMAGE_SIZE = 224
IMAGE_WIDTH, IMAGE_HEIGHT = IMAGE_SIZE, IMAGE_SIZE
EPOCHS = 5
BATCH_SIZE = 16

input_shape = (IMAGE_WIDTH, IMAGE_HEIGHT, 3)

model = Sequential()

model.add(Conv2D(32, (3, 3), border_mode='same', input_shape=input_shape, activation='relu'))
model.add(Conv2D(32, (3, 3), border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (1, 1), border_mode='same', activation='relu'))
model.add(Conv2D(64, (1, 1), border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), border_mode='same', activation='relu'))
model.add(Conv2D(128, (3, 3), border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (2, 2), border_mode='same', activation='relu'))
model.add(Conv2D(256, (2, 2), border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(8, activation='relu'))
model.add(Dense(3))
    
model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])
# Restore the weights
model.load_weights('saved_models/trained_weights')



path = 'data/test/COVID-19/nejmoa2001191_f4.jpeg'

img = cv2.imread(path,-1)
plt.imshow(img)
img = image.load_img(path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
#result = []
images = np.vstack([x])
classes = list(model.predict(images))

print("Seems to be a ", end='')

if max(classes[0]) == classes[0][0]:
    print("COVID-19 Case")
elif max(classes[0])== classes[0][1]:
    print("NORMAL Case")
elif max(classes[0])== classes[0][2]:
    print("PNEUMONIA Case")
