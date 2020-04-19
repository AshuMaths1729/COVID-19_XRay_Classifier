import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import RMSprop, Adam
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import plot_model
import matplotlib.pyplot as plt
import os

MODEL_FILE = "trained_model.h5"
SAVE_DIR = os.path.join(os.getcwd(), 'saved_models')

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


#plot_model(model, to_file='Model.png')

# Data augmentation
training_data_generator = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True)
validation_data_generator = ImageDataGenerator(rescale=1./255)
test_data_generator = ImageDataGenerator(rescale=1./255)

# Data preparation
training_generator = training_data_generator.flow_from_directory(
    training_data_dir,
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=BATCH_SIZE)

val_generator = validation_data_generator.flow_from_directory(
    valid_data_dir,
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=1,
    shuffle=False)

# Training
history = model.fit_generator(
            training_generator,
            steps_per_epoch=len(training_generator.filenames) // BATCH_SIZE,
            epochs=EPOCHS,
            verbose=1,
            validation_data=val_generator)

# Save model and weights
if not os.path.isdir(SAVE_DIR):
    os.makedirs(SAVE_DIR)
model_path = os.path.join(SAVE_DIR, MODEL_FILE)
model.save(model_path)
print('Saved trained model at %s '% MODEL_FILE)


## Plot Training stats
# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('Accuracy_Plot.png')

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('Loss_Plot.png')
