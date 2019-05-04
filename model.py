import os
import csv
import cv2
import keras
import sklearn
import numpy as np
from scipy import ndimage
from random import shuffle
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from sklearn.model_selection import train_test_split
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D, Dropout


rate = 0.25
batch_size = 64
samples = []
#read in csv data points for all camera images and steering angles
with open('../CarND-Behavioral-Cloning-P3/data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		samples.append(line)

shuffle(samples) # randomize the loaded data points
#creates training dataset 80% , and validation dataset 20%  
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
	
#Generator function produces batches of the datasets to be fed to the model.
#it reduces the memory requirements by providing data points on demand.
#The data in the batches is augmented by flipping all camera images and adding
#steering angles to the left and right cmaera images. THese images are added
#to provide the corrective effect to return to the center of track when car is off center 
def generator(samples, batch_size):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            clr_images = []
            steer_angles = []
            for batch_sample in batch_samples:
                i = 0
                steer_adj = [0.0, 0.4, -0.4] #center, left and right images steering angle adjustment

                while i < 3:
                    fname = '../CarND-Behavioral-Cloning-P3/data/IMG/'+batch_sample[i].split('/')[-1]
                    image = ndimage.imread(fname) #read in image as RGB 
                    steer_angle = float(batch_sample[3]) + steer_adj[i]
                    clr_images.append(image)
                    steer_angles.append(steer_angle)
                    clr_images.append(cv2.flip(image,1)) #flips image along its vertical axis
                    steer_angles.append(steer_angle * -1.0) #negates the steering angle for the reversed image
                    i+=1

            X_train = np.array(clr_images)
            y_train = np.array(steer_angles)
            yield sklearn.utils.shuffle(X_train, y_train) #releases batches of data points unlike return that produces the entire data set


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size)
validation_generator = generator(validation_samples, batch_size)

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
# trim image to only see section with road
model.add(Cropping2D(cropping=((70,25), (0,0))))
#covnet layers executes increasingly complex pattern recognition on images
model.add(Conv2D(24,(5,5), strides=(2,2), activation='relu'))
# dropout drops some data flows flowing through the model to reduce overfiiting to the data
model.add(Dropout(rate)) 
model.add(Conv2D(36,(5,5), strides=(2,2), activation='relu'))
model.add(Dropout(rate))
model.add(Conv2D(48,(5,5), strides=(2,2), activation='relu'))
model.add(Dropout(rate))
model.add(Conv2D(64,(3,3), activation='relu'))
model.add(Conv2D(64,(3,3), activation='relu'))
model.add(Dropout(rate))
model.add(Flatten())
model.add(Dense(100)) 
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1)) #fully connected layers produces the steering angles as output

model.compile(loss='mse', optimizer='adam')

#calls are made to the batch generator for training and validation data points
#the history_object variable stores the  training and validation loss for each epoch
history_object = model.fit_generator(train_generator, steps_per_epoch =
len(train_samples)//batch_size, validation_data = 
validation_generator,
validation_steps = len(validation_samples)//batch_size, 
epochs=5, verbose=1)

model.save('model.h5')

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
exit()
