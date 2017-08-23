import csv
import cv2
import numpy as np


def process_image(image, measurement):
#     image = np.asarray(image)
    image_f = cv2.flip(image, 1)
    return ((image, measurement), (image_f, -measurement))
    
lines = []
with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
images = []
measurements = []
correction = 0.2

for row in lines[1:]:
    steering_center = float(row[3])
    source_path = line[0]

    steering_left = steering_center + correction
    steering_right = steering_center - correction

    filename = source_path.split('/')[-1]
    for name, steer in zip(row[:3], [steering_center, steering_left, steering_right]):
        file_path = '../data/' + name.strip()
        left, right = process_image(cv2.imread(file_path), steer)
        images.extend([np.asarray(left[0]), np.asarray(right[0])])
        measurements.extend([left[1], right[1]])
        


X_train = np.array(images)
y_train = np.array(measurements)


print("Shape of X_train is:", X_train.shape)
print("Shape of y_train is:", y_train.shape)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D
from keras.layers import Cropping2D
from keras.models import Model
import matplotlib.pyplot as plt

model = Sequential()
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(65, 320, 3)))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history = model.fit(X_train, y_train, validation_split=0.2, shuffle=True)

model.save('model.h5')


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
