import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import cv2

train_df = pd.read_csv(r"C:\Users\user\Desktop\ECG_Project\MIT-BIH\mitbih_train.csv", header = None)

train_X = train_df.drop(187, axis = 1)
train_Y = train_df[187]
train_Y = train_Y.to_numpy()
del train_df

x = []
print(train_X.shape)
for i in range(187):
    x.append(i)

original_img = np.load('train_image.npy')

append_img = np.zeros((187,187))


plt.figure()

i = 87553

plt.plot(x, train_X.loc[i,:])
plt.savefig("temp_pulse.png")
plt.clf()

img = cv2.imread("temp_pulse.png", cv2.IMREAD_GRAYSCALE)
# print(type(img))
img = img[61:424, 82:574]
img = cv2.resize(img, (187,187))

append_img = img/256.0
print(original_img.shape, append_img.shape)

train_image = np.append(original_img, [append_img], axis = 0)

print(train_image.shape)
np.save("train_image_new.npy", train_image)

exit()

with open("train_image.npy", 'wb') as pickle_data:
    np.save("train_image.npy", train_image)

with open("test_image.npy", 'wb') as pickle_data:
    np.save("test_image.npy", test_image)


with open("train_label.npy", 'wb') as pickle_data:
    np.save("train_label.npy", train_Y)

with open("test_label.npy", 'wb') as pickle_data:
    np.save("test_label.npy", test_Y)
