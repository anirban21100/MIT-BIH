from tensorflow import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization
from sklearn.metrics import confusion_matrix
import seaborn as sns
from collections import Counter
from matplotlib import pyplot as plt
import tensorflow as tf
import datetime as dt

# Load Data
train_X = np.load(r"C:\Users\user\Desktop\ECG_Project\MIT-BIH\mitbih_processed_image\train_image_shuffled.npy", mmap_mode='r')
val_X = np.load(r"C:\Users\user\Desktop\ECG_Project\MIT-BIH\mitbih_processed_image\validation_image.npy")
train_Y = np.load(r"C:\Users\user\Desktop\ECG_Project\MIT-BIH\mitbih_processed_image\train_label_shuffled.npy")
val_Y = np.load(r"C:\Users\user\Desktop\ECG_Project\MIT-BIH\mitbih_processed_image\validation_label.npy")


# from_generator
def generator():
    for i in range(train_Y.shape[0]):
        yield (train_X[i], train_Y[i])

dataset = tf.data.Dataset.from_generator(generator, (tf.float64,tf.float32), ((187,187,1), (5)))

dataset = dataset.batch(32) 

for i in range(10):
    # Model (MNIST model에 Conv2D 64짜리 추가)
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

    model.add(Flatten())

    model.add(Dense(120, activation='relu'))

    model.add(Dense(5, activation = 'softmax'))

    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    # Callbacks
    early_stopping_cb = keras.callbacks.EarlyStopping(patience = 10, restore_best_weights = True) 
    checkpoint_cb = keras.callbacks.ModelCheckpoint("MIT_BIH_"+str(i)+"1.h5", save_best_only = True)

    # Imbalaced, {0: 72471, 1: 2223, 2: 5788, 3: 641, 4: 6431} -> class weight (cost-sensitive)

    weights = {0:1, 1:32.601, 2:12.521, 3:113.059, 4:11.269}

    history = model.fit(dataset, epochs = 10000, steps_per_epoch = 2456, class_weight = weights, validation_data=(val_X, val_Y),\
        callbacks=[early_stopping_cb, checkpoint_cb])


exit()
# Free Resource
del train_X
del val_X

test_X = np.load(r"C:\Users\user\Desktop\ECG_Project\MIT-BIH\mitbih_image\test_image.npy")
test_Y = np.load(r"C:\Users\user\Desktop\ECG_Project\MIT-BIH\mitbih_image\test_label.npy")


# model evaluate method
eval = model.evaluate(test_X, test_Y)
print(eval)

# Get Predictions (reverse to_categorical)
pred_proba = model.predict(test_X)
prediction = np.argmax(pred_proba, axis = 1)
actual = np.argmax(test_Y, axis = 1)

correct = 0
wrong_indices = []
wrong_actual = []
wrong_pred = []

for i in range(actual.shape[0]):
    if actual[i] == prediction[i]:
        correct += 1
    else:
        wrong_indices.append(i)
        wrong_pred.append(prediction[i])
        wrong_actual.append(actual[i])


accuracy = correct/int(actual.shape[0])
print("Test Accuracy = ", accuracy)


cf_matrix = confusion_matrix(actual, prediction)
cf_matrix = sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, fmt='.2%', cmap='Blues')
plt.savefig("confusion matrix.jpg")

print(dict(Counter(wrong_actual)))