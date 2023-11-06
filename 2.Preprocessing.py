import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import collections
import random
from keras.utils.np_utils import to_categorical

original_train_image = np.load(r'train_image.npy')
original_test_image = np.load(r'test_image.npy')
original_train_Y = np.load(r'train_label.npy')
original_test_Y = np.load(r'test_label.npy')

print(original_train_image.shape, original_train_Y.shape)
print(original_test_image.shape, original_test_Y.shape)

# plt.imshow(original_train_image[0])

train_Y_full = original_train_Y.astype('int8')
test_Y = original_test_Y.astype('int8')

# Checking balance between labels
train_count = dict(collections.Counter(train_Y_full))
print(train_count)
print(train_Y_full.shape[0])
test_count = dict(collections.Counter(test_Y))

x1 = list(train_count.keys())
x2 = list(test_count.keys())
y1 = list(train_count.values())
y2 = list(test_count.values())

fig, (ax1, ax2) = plt.subplots(1,2)
plt.rcParams["figure.figsize"] = (10,5)
fig.suptitle('MIT-BIH label')
ax1.pie(y1, labels = x1, autopct='%1.1f%%', shadow=True, startangle=90)
ax1.axis('equal')

ax2.pie(y2, labels = x2, autopct='%1.1f%%', shadow=True, startangle=90)
ax2.axis('equal')

plt.show()
# Highly imbalanced.

# Preprocessing

#1.Add third channel
# Channel
print(original_train_image.shape, original_test_image.shape)
train_X_full = original_train_image.reshape(-1, 187, 187, 1) 
del original_train_image # free resource
test_X = original_test_image.reshape(-1, 187, 187, 1)
del original_test_image
print(train_X_full.shape, test_X.shape)

#2. Split Data
# Data Split
df = pd.DataFrame(train_Y_full, columns = ['label'])
print(df.value_counts())

label0 = df.index[df['label'] ==0].tolist()
label1 = df.index[df['label'] ==1].tolist()
label2 = df.index[df['label'] ==2].tolist()
label3 = df.index[df['label'] ==3].tolist()
label4 = df.index[df['label'] ==4].tolist()

val_0 = random.sample(label0, 7452)
val_1 = random.sample(label1, 225)
val_2 = random.sample(label2, 594)
val_3 = random.sample(label3, 63)
val_4 = random.sample(label4, 657)
del df
val_index = val_0 + val_1 + val_2 + val_3 + val_4
print(len(val_index))

train_X = np.delete(train_X_full, val_index, axis = 0)
val_X = train_X_full[val_index]
del train_X_full

train_Y = np.delete(train_Y_full, val_index, axis = 0)
val_Y = train_Y_full[val_index]
del train_Y_full

print(val_X.shape, val_Y.shape,train_X.shape, train_Y.shape)

# 3. Zero Center
zero_centerer = np.mean(train_X, axis = 0)
train_X -= zero_centerer
val_X -= zero_centerer
test_X -= zero_centerer

# 4. One hot encoding

train_Y = to_categorical(train_Y, num_classes = 5)
val_Y = to_categorical(val_Y, num_classes = 5)
test_Y = to_categorical(test_Y, num_classes = 5)

# Save Data
np.save(r"train_image.npy", train_X)
np.save(r"validation_image.npy", val_X)
np.save(r"test_image.npy", test_X)
np.save(r"train_label.npy",train_Y)
np.save(r"validation_label.npy", val_Y)
np.save(r"test_label.npy", test_Y)
np.save(r"zero_centerer.npy", zero_centerer)
