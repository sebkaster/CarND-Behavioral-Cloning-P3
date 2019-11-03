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
                left_angle = float(line[3])
                angles.append(left_angle + correction)
                image_flipped = np.fliplr(center_left)
                measurement_flipped = -left_angle
                images.append(image_flipped)
                angles.append(measurement_flipped)

                current_path = 'data/IMG/' + batch_sample[2].split('/')[-1]
                center_right = preprocess_img(cv2.imread(current_path))
                images.append(center_right)
                right_angle = float(line[3])
                angles.append(right_angle - correction)
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

model = tf.keras.Sequential()
model.add(tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32) / 127.5 - 1, input_shape=(90, 320, 3)))

model.add(tf.keras.layers.Conv2D(6, 5, 5, activation='relu'))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(6, 5, 5, activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(padding='same'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(120))
model.add(tf.keras.layers.Dense(84))
model.add(tf.keras.layers.Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, steps_per_epoch=ceil(len(train_samples) / batch_size),
                    validation_data=validation_generator, validation_steps=ceil(len(validation_samples) / batch_size),
                    epochs=5)

model.save('model.h5')
