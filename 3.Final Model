from tensorflow import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization
from sklearn.metrics import confusion_matrix
import seaborn as sns
from collections import Counter
from matplotlib import pyplot as plt
import tensorflow as tf

# Load Data
train_X = np.load(r"train_image.npy", mmap_mode='r')
val_X = np.load(r"validation_image.npy")
train_Y = np.load(r"train_label.npy")
val_Y = np.load(r"validation_label.npy")


# from_generator
def generator():
    for i in range(train_Y.shape[0]):
        yield (train_X[i], train_Y[i]) 

# Using from_generator as the dataset is too big for RAM
dataset = tf.data.Dataset.from_generator(generator, (tf.float64,tf.float32), ((187,187,1), (5)))

dataset = dataset.batch(32) 


### Modification History ###
'''
*** optimizer : adam, loss function : categorical crossentropy, activation : relu
Base architecture : conv2D (3x3, 32) - conv2D (3x3, 32) - conv2D (5x5, 64) - conv2D (5x5, 64) - conv2D (5x5, 128)- conv2D (5x5, 128)
    * weight : {0: 1, 1: 32.601, 2: 12.521, 3: 113.059, 4: 11.269}

modification1 : change weight : * weight : {0: 1, 1: 50, 2: 12.521, 3: 100, 4: 11.269} -> update class 1 weigth back to 32.601, class 3 weight updated to 100
modification2 : change depth (without changing weight) : conv2D (3x3, 32) - conv2D (3x3, 32) - conv2D (5x5, 64) - conv2D (5x5, 64) - conv2D (5x5, 128)- conv2D (5x5, 128) - conv2D (5x5, 128) (added conv2D128)
    * weight : {0: 1, 1: 32.601, 2: 12.521, 3: 113.059, 4: 11.269}
modification3 : added dropout : performance drop - eliminated
    * weight : {0: 1, 1: 32.601, 2: 12.521, 3: 113.059, 4: 11.269}
modification4 : weight change + depth change  (modification 1 + modification 2) - conv2D (3x3, 32) - conv2D (3x3, 32) - conv2D (5x5, 64) - conv2D (5x5, 64) - conv2D (5x5, 128)- conv2D (5x5, 128) - conv2D (5x5, 128)
    * weight : {0: 1, 1: 32.601, 2: 12.521, 3: 100, 4: 11.269}
modification5 : added more depth : performance drop - eliminated

FINAL MODEL : modification 4 + 10 model ensemble
'''
for i in range(10):

    model =Sequential()
    model.add(Conv2D(32, (3,3), activation='relu', input_shape=(187,187,1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())

    model.add(Conv2D(32, (3,3), activation='relu')) 
    model.add(BatchNormalization())
    model.add(MaxPooling2D())

    model.add(Conv2D(64, (5,5), activation='relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())

    model.add(Conv2D(64, (5,5), activation='relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())

    model.add(Conv2D(128, (5,5), activation='relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())

    model.add(Conv2D(128, (5,5), activation='relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())

    model.add(Conv2D(128, (5,5), activation='relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())

    model.add(Flatten())

    model.add(Dense(120, activation='relu'))

    model.add(Dense(5, activation = 'softmax'))

    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    # Callbacks
    early_stopping_cb = keras.callbacks.EarlyStopping(patience = 10, restore_best_weights = True) 
    checkpoint_cb = keras.callbacks.ModelCheckpoint("deeper_changed_weight_MIT_BIH_V"+str(i)+".h5", save_best_only = True)

    # Imbalaced, {0: 72471, 1: 2223, 2: 5788, 3: 641, 4: 6431} -> class weight (cost-sensitive)
    weights = {0:1, 1:32.601, 2:12.521, 3:100, 4:11.269}

    history = model.fit(dataset.repeat(), epochs = 10000, steps_per_epoch = 2456, class_weight = weights, validation_data=(val_X, val_Y),\
        callbacks=[early_stopping_cb, checkpoint_cb])
