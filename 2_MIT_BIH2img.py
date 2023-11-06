import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import cv2

train_df = pd.read_csv(r"C:\Users\user\Desktop\ECG_Project\MIT-BIH\mitbih_csv\mitbih_train.csv", header = None)
test_df = pd.read_csv(r"C:\Users\user\Desktop\ECG_Project\MIT-BIH\mitbih_csv\mitbih_test.csv", header = None)


print(train_df.head(3))
print(train_df.info())
print(train_df[187].value_counts())
train_X = train_df.drop(187, axis = 1)
train_Y = train_df[187]
train_Y = train_Y.to_numpy()
print(train_Y.dtype)

print(test_df.info())
test_X = test_df.drop(187, axis = 1)
test_Y = test_df[187]
test_Y = test_Y.to_numpy()

x = []
print(train_X.shape)
for i in range(187):
    x.append(i)

train_image = np.zeros((87553, 187, 187))
test_image = np.zeros((21892, 187, 187))


plt.figure()


for i in range(0, 87554):
    if i % 100 == 0:
        print(i)
    plt.plot(x, train_X.loc[i,:])
    plt.savefig("test_temp_pulse.png")
    plt.clf()
    print(plt.rcParams["figure.figsize"])
    img = cv2.imread("test_temp_pulse.png", cv2.IMREAD_GRAYSCALE)
    print(img.shape)
    img = img[61:424, 82:574]
    img = cv2.resize(img, (187,187))

    # train_image[i] = img/256.0
    if i == 0:
        break
    


for i in range(0, 21892):
    if i % 100 == 0:
        print(i)
    plt.plot(x, test_X.loc[i,:])
    plt.savefig("temp_pulse_test.png")
    plt.clf()

    img = cv2.imread("temp_pulse_test.png", cv2.IMREAD_GRAYSCALE)

    img = img[61:424, 82:574]
    img = cv2.resize(img, (187,187))

    test_image[i] = img/256.0


np.save("train_image.npy", train_image)

np.save("test_image.npy", test_image)


np.save("train_label.npy", train_Y)

np.save("test_label.npy", test_Y)