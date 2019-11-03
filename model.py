import csv
import cv2
import numpy as np
import tensorflow as tf
from math import ceil

import sklearn
from sklearn.model_selection import train_test_split

from preprocess import preprocess_img
from random import shuffle

import matplotlib.pyplot as plt
import keras
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices)

lines = []
with open('data/driving_log.csv') as csvfile:
    csvfile.readline()  # skip the first line
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

train_samples, validation_samples = train_test_split(lines, test_size=0.2)

correction = 0.25


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while (1):  # loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []

            for batch_sample in batch_samples:
                current_path = 'data/IMG/' + batch_sample[0].split('/')[-1]
                center_image = preprocess_img(cv2.imread(current_path))
                images.append(center_image)
                center_angle = float(line[3])
                angles.append(center_angle)
                image_flipped = np.fliplr(center_image)
                measurement_flipped = -center_angle
                images.append(image_flipped)
                angles.append(measurement_flipped)

                current_path = 'data/IMG/' + batch_sample[1].split('/')[-1]
                center_left = preprocess_img(cv2.imread(current_path))
                images.append(center_left)
                left_angle = float(line[3]) + correction
                angles.append(left_angle)
                image_flipped = np.fliplr(center_left)
                measurement_flipped = -left_angle
                images.append(image_flipped)
                angles.append(measurement_flipped)

                current_path = 'data/IMG/' + batch_sample[2].split('/')[-1]
                center_right = preprocess_img(cv2.imread(current_path))
                images.append(center_right)
                right_angle = float(line[3]) - correction
                angles.append(right_angle)
                image_flipped = np.fliplr(center_right)
                measurement_flipped = -right_angle
                images.append(image_flipped)
                angles.append(measurement_flipped)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield (X_train, y_train)


batch_size = 32

train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

model = keras.Sequential()
model.add(keras.layers.Lambda(lambda x: x / 127.5 - 1, input_shape=(90, 320, 3)))

model.add(
    keras.layers.Conv2D(24, kernel_size=5, strides=2, activation='elu', padding='valid', kernel_regularizer=keras.regularizers.l2(0.001)))
model.add(
    keras.layers.Conv2D(36, kernel_size=5, strides=2, activation='elu', padding='valid', kernel_regularizer=keras.regularizers.l2(0.001)))
model.add(
    keras.layers.Conv2D(48, kernel_size=5, strides=2, activation='elu', padding='same', kernel_regularizer=keras.regularizers.l2(0.001)))

model.add(
    keras.layers.Conv2D(64, kernel_size=3, activation='elu', padding='same', kernel_regularizer=keras.regularizers.l2(0.001)))
model.add(
    keras.layers.Conv2D(64, kernel_size=3, activation='elu', padding='same', kernel_regularizer=keras.regularizers.l2(0.001)))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100, kernel_regularizer=keras.regularizers.l2(0.001), activation='elu'))
model.add(keras.layers.Dense(50, kernel_regularizer=keras.regularizers.l2(0.001), activation='elu'))
model.add(keras.layers.Dense(10, kernel_regularizer=keras.regularizers.l2(0.001), activation='elu'))
model.add(keras.layers.Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, steps_per_epoch=ceil(len(train_samples) / batch_size),
                    validation_data=validation_generator, validation_steps=ceil(len(validation_samples) / batch_size),
                    epochs=5)

model.save('model.h5')
