import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import cv2

train_df = pd.read_csv(r"C:\Users\user\Desktop\ECG_Project\MIT-BIH\mitbih_train.csv", header = None)
test_df = pd.read_csv(r"C:\Users\user\Desktop\ECG_Project\MIT-BIH\mitbih_test.csv", header = None)


print(train_df.head(3))
print(train_df.info())
print(train_df[187].value_counts())
train_X = train_df.drop(187, axis = 1)
train_Y = train_df[187]
train_Y = train_Y.to_numpy()

print(test_df.info())
test_X = test_df.drop(187, axis = 1)
test_Y = test_df[187]
test_Y = test_Y.to_numpy()

x = []
print(train_X.shape)
for i in range(187):
    x.append(i)

train_image = np.zeros((87553, 60, 60))
test_image = np.zeros((21892, 60, 60))


plt.figure()

import time
for i in range(0, 87554):
    if i % 100 == 0:
        print(i)
    plt.plot(x, train_X.loc[i,:])
    plt.savefig("temp_pulse.png")
    plt.clf()

    img = cv2.imread("temp_pulse.png", cv2.IMREAD_GRAYSCALE)
    # print(type(img))
    img = img[61:424, 82:574]
    img = cv2.resize(img, (60,60))

    train_image[i] = img/256.0
    
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("teset.jpg", img)
    time.sleep(3)


exit()
for i in range(0, 21892):
    if i % 100 == 0:
        print(i)
    plt.plot(x, test_X.loc[i,:])
    plt.savefig("temp_pulse_test.png")
    plt.clf()

    img = cv2.imread("temp_pulse_test.png", cv2.IMREAD_GRAYSCALE)

    img = img[61:424, 82:574]
    img = cv2.resize(img, (60,60))

    test_image[i] = img/256.0


exit()
# import pickle

np.save("train_image.npy", train_image)

np.save("test_image.npy", test_image)


np.save("train_label.npy", train_Y)

np.save("test_label.npy", test_Y)

exit()

with open("train_image.npy", 'wb') as pickle_data:
    np.save("train_image.npy", train_image)

with open("test_image.npy", 'wb') as pickle_data:
    np.save("test_image.npy", test_image)


with open("train_label.npy", 'wb') as pickle_data:
    np.save("train_label.npy", train_Y)

with open("test_label.npy", 'wb') as pickle_data:
    np.save("test_label.npy", test_Y)
