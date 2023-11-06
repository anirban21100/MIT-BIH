import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

original_train_image = np.load(r'C:\Users\user\Desktop\ECG_Project\train_image_new.npy')
original_test_image = np.load(r'C:\Users\user\Desktop\ECG_Project\test_image.npy')
original_train_Y = np.load(r'C:\Users\user\Desktop\ECG_Project\train_label.npy')
original_test_Y = np.load(r'C:\Users\user\Desktop\ECG_Project\test_label.npy')
