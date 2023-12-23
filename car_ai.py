# Project: CNN Nvidia 학습 모델 생성하기(2단계: 모델 생성)
# Filename: car_ai.py
# Created Date: 2023-12-08(금)
# Author: 대학원생 석사과정 정도윤
# Description:
# 1. tensorflow 2.4버전 이상으로 형식 변경
# 2. 
#
# Reference:
# 1. https://github.com/aruneshmee/Self-Driving-Car, Accessed by 2023-12-23.
# 2. AI 인공지능 자율주행 자동차 만들기 + 데이터 수집·학습 + 딥러닝 with 라즈베리파이, 장문철, 앤써북, 2021-08-30.
#
import os
from pickletools import optimize
import random
import fnmatch
import datetime
import pickle



# data processing
import numpy as np
np.set_printoptions(formatter={'float_kind':lambda x: "%.4f" % x})

import pandas as pd
pd.set_option('display.width', 300)
pd.set_option('display.float_format', '{:,.4f}'.format)
pd.set_option('display.max_colwidth', 200)


# tensorflow
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.models import load_model


print( f'tf.__version__: {tf.__version__}')
print( f'.keras__version__:{keras.__version__}')


# sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


# imaging
import cv2
from imgaug import augmenters as img_aug
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# %matplotlib inline
# from PIL import image
from PIL import Image

data_dir = 'D:/python_opencv/self_driving_car/video'
file_list = os.listdir(data_dir)
image_paths = []
steering_angles = []
pattern = "*.png"

for filename in file_list:
    if fnmatch.fnmatch(filename, pattern):
        image_paths.append(os.path.join(data_dir, filename))
        angle = int(filename[-7:-4])
        steering_angles.append(angle)

image_index = 20
plt.imshow(Image.open(image_paths[image_index]))
print("image_path: %s" % image_paths[image_index])
print("steering_Angle: %d" % steering_angles[image_index])
df = pd.DataFrame()
df['ImagePath'] = image_paths
df['Angle'] = steering_angles


num_of_bins = 25
hist, bins = np.histogram(df['Angle'], num_of_bins)


fig, axes = plt.subplots(1, 1, figsize=(12, 4))
axes.hist(df['Angle'], bins=num_of_bins, width=1, color='blue')


X_train, X_valid, y_train, y_valid = train_test_split( image_paths, steering_angles, test_size=0.2)
print("Training data: %d\nValidation data: %d" % (len(X_train), len(X_valid)))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(y_train, bins=num_of_bins, width=1, color='blue')
axes[0].set_title('Training Data')
axes[1].hist(y_valid, bins=num_of_bins, width=1, color='red')
axes[1].set_title('Validation Data')

def my_imread(image_path):
    image = cv2.imread(image_path)
    return image

def img_preprocess(image):
    image = image / 255
    return image


fig, axes = plt.subplots(1, 2, figsize=(15, 10))
image_orig = my_imread(image_paths[image_index])
image_processed = img_preprocess(image_orig)
axes[0].imshow(image_orig)
axes[0].set_title("orig")
axes[1].imshow(image_processed)
axes[1].set_title("processed")


def nvidia_model():
    model = Sequential(name='Nvidia_Model')

    model.add(Conv2D(24, (5, 5), strides=(2,2), input_shape=(120, 400, 3), activation='relu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu'))

    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))

    model.add(Dense(1))
    
    optimizer = Adam(lr=1e-3)
    model.compile(loss='mse', optimizer=optimizer)

    return model

model = nvidia_model()
print(model.summary())


# 학습 데이터 생성
def image_data_generator(image_paths, steering_angles, batch_size):
    while True:
        batch_images = []
        batch_steering_angles = []

        for i in range(batch_size):
            random_index = random.randint(0, len(image_paths) - 1)
            image_path = image_paths[random_index]
            image = my_imread(image_paths[random_index])
            steering_angle = steering_angles[random_index]


            image = img_preprocess(image)
            batch_images.append(image)
            batch_steering_angles.append(steering_angle)

        yield(np.asarray(batch_images), np.asarray(batch_steering_angles))


ncol = 2
nrow = 2


X_train_batch, y_train_batch = next(image_data_generator(X_train, y_train, nrow))
X_valid_batch, y_valid_batch = next(image_data_generator(X_valid, y_valid, nrow))


fig, axes = plt.subplots(nrow, ncol, figsize=(15, 6))
fig.tight_layout()


for i in range(nrow):
    axes[i][0].imshow(X_train_batch[i])
    axes[i][0].set_title("training, angle = %s" % y_train_batch[i])
    axes[i][1].imshow(X_valid_batch[i])
    axes[i][1].set_title("validation, angle = %s" % y_valid_batch[i])

    model_output_dir = "D:/python_opencv/self_driving_car/output_model"

    checkpoint_callback = ModelCheckpoint(filepath=os.path.join(model_output_dir, '231207_lane_navigation_check.h5'), 
            verbose=1, save_best_only=True)

    history = model.fit(image_data_generator(X_train, y_train, batch_size=100),
                    steps_per_epoch=300, epochs=10, validation_data=image_data_generator(X_valid, y_valid, batch_size=100), validation_steps=200, verbose=1, shuffle=1, callbacks=[checkpoint_callback])


    model.save(os.path.join(model_output_dir, '231207_lane_navigation_final.h5'))

    history_path = os.path.join(model_output_dir, 'history.pickle')
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f ,pickle.HIGHEST_PROTOCOL)