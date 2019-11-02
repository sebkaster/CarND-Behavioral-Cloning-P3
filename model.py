import csv
import cv2
import numpy as np

lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
i = 0
for line in lines:
    if i == 0:
        i += 1
        continue
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = 'data/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)

import tensorflow as tf

model = tf.keras.Sequential()
model.add(tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32) / 255.0 - 0.5, input_shape=(160, 320, 3)))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=1)

model.save('model.h5')


