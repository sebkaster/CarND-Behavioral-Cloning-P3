import keras
import cv2
import numpy as np
import pandas as pd
import os
from math import ceil
from preprocess import img_crop, img_normalization


def random_brightness(img):
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2HSV)
    random_bright = 0.25 + np.random.uniform()
    img[:, :, 2] = img[:, :, 2] * random_bright
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    return img


def img_flip(img, angle):
    img = cv2.flip(img, 1)
    return img, -1.0 * angle


def image_generator(data, validation_flag):
    data = data.sample(frac=1).reset_index(drop=True)

    for index, row in driving_log.iterrows():

        # Select Left,Center,Right image
        select_camera_image = np.random.randint(3)

        if select_camera_image == 0:
            fname = os.path.basename(row['left'])
            steering = np.float32(row['steering']) + 0.25
        elif select_camera_image == 1:
            fname = os.path.basename(row['center'])
            steering = np.float32(row['steering'])
        else:
            fname = os.path.basename(row['right'])
            steering = np.float32(row['steering']) - 0.25

        img = keras.preprocessing.image.load_img('./data/IMG/' + fname)
        img = np.array(img)

        # Crop and Resize the image
        img = img_crop(img)
        # Normalize the image
        img = img_normalization(img)

        if np.random.randint(0, 1):
            # Add Random Brightness
            img = random_brightness(img)

        if np.random.randint(0, 1):
            # Flip image
            img, steering = img_flip(img, steering)

        # Change the color space
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2YUV)

        # Reshape the image
        img = np.reshape(img, (3, 66, 200))

        yield img, steering


def batch_generator(driving_log, validation_flag=False, batch_size=32):
    num_rows = len(driving_log.index)
    train_images = np.zeros((batch_size, 3, 66, 200))
    train_steering = np.zeros(batch_size)
    line_num = 0
    while True:
        for j in range(batch_size):
            # Reset generator if over bounds
            if line_num >= num_rows:
                line_num = 0
                images = image_generator(driving_log, validation_flag)
            elif line_num == 0:
                images = image_generator(driving_log, validation_flag)
            train_images[j], train_steering[j] = next(images)
            line_num += 1
        yield train_images, train_steering


# Cut off 75% of low steering angle
def remove_low_angles(driving_log, angle_threshold=0.1):
    num_drops = int(len(driving_log[np.abs(driving_log["steering"]) <= angle_threshold]) * 0.75)
    drop_lows = driving_log[driving_log["steering"] == 0]["index"].values[0:num_drops]

    return driving_log.drop(drop_lows, axis=0).sample(frac=1.0)


driving_log = pd.read_csv("./data/driving_log.csv").reset_index()
print("Number of Original Data", len(driving_log))

revised_log = remove_low_angles(driving_log)
print("Number of Revised Data", len(revised_log))

num_training = (int(len(revised_log) * 0.8))

training_data = revised_log[0:num_training]

print("Num of Training data", len(training_data))
validation_data = revised_log[num_training:]
print("Num of Validation data", len(validation_data))

# Make dataset
train_data = batch_generator(training_data)
val_data = batch_generator(validation_data, validation_flag=True)

model = keras.Sequential()

model.add(
    keras.layers.Conv2D(24, kernel_size=5, strides=2, activation='elu', padding='same',
                        kernel_regularizer=keras.regularizers.l2(0.001), input_shape=(3, 66, 200)))
model.add(
    keras.layers.Conv2D(36, kernel_size=5, strides=2, activation='elu', padding='same',
                        kernel_regularizer=keras.regularizers.l2(0.001)))
model.add(
    keras.layers.Conv2D(48, kernel_size=5, strides=2, activation='elu', padding='same',
                        kernel_regularizer=keras.regularizers.l2(0.001)))
model.add(
    keras.layers.Conv2D(64, kernel_size=3, activation='elu', padding='same',
                        kernel_regularizer=keras.regularizers.l2(0.001)))
model.add(
    keras.layers.Conv2D(64, kernel_size=3, activation='elu', padding='same',
                        kernel_regularizer=keras.regularizers.l2(0.001)))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100, kernel_regularizer=keras.regularizers.l2(0.001), activation='elu'))
model.add(keras.layers.Dense(50, kernel_regularizer=keras.regularizers.l2(0.001), activation='elu'))
model.add(keras.layers.Dense(10, kernel_regularizer=keras.regularizers.l2(0.001), activation='elu'))
model.add(keras.layers.Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_data, steps_per_epoch=ceil(len(training_data) / 32),
                    validation_data=val_data, validation_steps=ceil(len(validation_data) / 32),
                    epochs=10)

model.save('model.h5')
