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

# Load Test Data
test_X = np.load(r"C:\Users\user\Desktop\ECG_Project\MIT-BIH\mitbih_processed_image\train_image.npy")
test_Y = np.load(r"C:\Users\user\Desktop\ECG_Project\MIT-BIH\mitbih_processed_image\train_label.npy")

# Load Model
model = keras.models.load_model(r"C:\Users\user\Desktop\ECG_Project\mitbih_models\MIT_BIH_V1.h5")

for i in range(10):
    pass

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