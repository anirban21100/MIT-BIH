from tensorflow import keras
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
from collections import Counter
from matplotlib import pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report

import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


model = []
for i in range(10):
    model.append(keras.models.load_model(r"deeper_changed_weight_MIT_BIH_V"+str(i)+".h5"))

################################ Evaluating on test data ################################
test_X = np.load(r"test_image.npy")
test_Y = np.load(r"test_label.npy")

# Get Predictions (reverse to_categorical)
pred_proba = np.zeros(test_Y.shape)
for i in range(10):
    pred_proba = pred_proba + model[i].predict(test_X)

prediction = np.argmax(pred_proba, axis = 1)
actual = np.argmax(test_Y, axis = 1)
del test_X
del test_Y

print(classification_report(actual, prediction))

# Confusion matrix
cnf_matrix = confusion_matrix(actual, prediction)
np.set_printoptions(precision=2)
plt.figure(figsize=(5, 5))
plot_confusion_matrix(cnf_matrix, classes=['N', 'S', 'V', 'F', 'Q'],normalize=True,
                      title='Confusion matrix, with normalization')
plt.show()


################################ Checking on training data ################################
train_X = np.load(r"train_image.npy", mmap_mode = 'r')
train_Y = np.load(r"train_label.npy")
prediction_list = []
pred_proba = np.zeros(train_Y.shape)
for i in range(train_Y.shape[0]):
    question = np.array([train_X[i]])
    for j in range(10):
        pred_proba = pred_proba + model[j].predict(question)
    answer = np.argmax(pred_proba, axis = 1)
    prediction_list.append(answer)

prediction = np.array(prediction_list)
actual = np.argmax(train_Y, axis = 1)

print(classification_report(actual, prediction))

# Compute confusion matrix
cnf_matrix = confusion_matrix(actual, prediction)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure(figsize=(5, 5))
plot_confusion_matrix(cnf_matrix, classes=['N', 'S', 'V', 'F', 'Q'],normalize=True,
                      title='Confusion matrix, with normalization')
plt.show()
