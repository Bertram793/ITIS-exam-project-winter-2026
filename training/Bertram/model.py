import tensorflow as tf
import os 
import matplotlib.pyplot as plt
from keras.utils import img_to_array, load_img
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping

train_path = "C:/DTU/ITIS/archive/train/train/"
test_path = "C:/DTU/ITIS/archive/test/test/"

Batch_size = 32

img = load_img(train_path + "Apple Braeburn/Apple Braeburn_0.jpg")

img1 = img_to_array(img)
print(img1.shape)

model = Sequential()
model.add(Conv2D(filters=64, kernel_size=3, activation='relu', input_shape=(100, 100, 3)))
model.add(MaxPooling2D())
model.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
model.add(Conv2D(filters=16, kernel_size=3, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(5000, activation='relu'))
model.add(Dense(1000, activation='relu'))
model.add(Dense(131, activation='softmax'))