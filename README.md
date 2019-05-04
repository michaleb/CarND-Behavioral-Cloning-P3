## **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results


[//]: # (Image References)

[image1]: ./examples/val_loss.png "Training / Validation Loss"
[image2]: ./examples/image.jpg "Center Camera Image"
[image3]: ./examples/left_image.jpg "Left Camera Image"
[image4]: ./examples/right_image.jpg "Right Camera Image"
[image5]: ./examples/flipped_image.jpg "Flipped Image"
[image6]: ./examples/center_image.jpg "Center Camera Image"

---
My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* video.mp4 captures one and a half laps of autonomous driving around track one.
* A summary of the results in the README

### Overview

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model.

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 63, 65, 67, 69 and 70). 

The model includes RELUs to introduce nonlinearity (code lines 63, 65, 67, 69 and 70), and the data is normalized in the model using a Keras lambda layer (code line 60) and dropout layers in order to reduce overfitting (model.py lines 64, 66, 68 and 71). 

Trained and validated was executed on different data sets to safeguard against overfitting (code line 55-56). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 78).

#### Training data

Training data was chosen to keep the vehicle driving on the road. I used only center lane driving and data augmentation to mimic recovering from the left and right sides of the road. Due to the layout of the track there is an inherent left turn bias so flipping the images will provide an equal number of right turns.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to look at existing models that demonstrated good performance on image data.

My first step was to use a convolution neural network model similar to the LeNet model. I thought this model might be appropriate because I had previously used it to detect images of traffic signs.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model, using LeNet and the given dataset,  had a low mean squared error on both the training and validation sets but the car did not make it past the bridge. 

I then captured images from three and a half laps of driving on track one and used images from all three cameras. The images were also flipped and the appropriate steer angles associated with them. However, the car ran off the road at the first deep left corner. I then changed to using Nvidia's model but observed a higher mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that some flows were dropped between covnet layers and to the fully connected layers then I reduced the number of epochs from ten as the validation loss was trending upwards after five epochs. I also increased the steer adjustments to provide quicker off track recovery.

![alt text][image1]

The final step was to run the simulator to see how well the car was driving around track one. There was a single spot where the vehicle went into the hatched marked area to the side of the track. To improve the driving behavior in this case, I increased the steering adjustment angle (recovery angles) for the offset left and right cameras.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 63-76) consisted of a convolution neural network with the following layers and layer sizes:
Layer 1 - Covnet - 24 @ 30x158
Layer 2 - Dropout - 25%
Layer 3 - Covnet - 36 @ 13x77
Layer 4 - Dropout - 25%
Layer 5 - Covnet - 48 @  4x36
Layer 6 - Dropout - 25%
Layer 7 - Covnet - 64 @  3x33
Layer 8 - Covnet - 64 @  1x30
Layer 9 - Flatten - 1920
Layer 10 - Dense - 100
Layer 11 - Dense - 50
Layer 12 - Dense - 10
Layer 13 - Dense 1 (output steer angle)


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three and a half laps on track one using center lane driving, my dexterity improved with successive laps. Here is an example image of center lane driving:

![alt text][image2]

All data was captured from track one.

To augment the data set, I included images from the left and right cameras and added a corresponding steering adjustments. The left and right images alongwith their steering adjustments capture what a recovery from the side of the road should look like. Here are two images one each from the left and right cameras.

![alt text][image3]

![alt text][image4]

In addition all images and their angles were flipped thinking that this would create the effect of driving in the clockwise direction thus removing any left turn biases caused by the layout of the track. For example, here is a center camera image and a flipped version of it.

![alt text][image6]

![alt text][image5]


After the collection process, I had 11,517 number of data points. I then preprocessed this data, using a Lambda layer in the model, by reducing all pixel values to between [-0.5, 0.5].


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the validation loss beginning to increase. I used an Adam optimizer so that manually training the learning rate wasn't necessary.

Here's a [link to my video result](/examples/video.mp4)


## Udacity's original README

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This repository contains starting files for the Behavioral Cloning Project.

In this project, you will use what you've learned about deep neural networks and convolutional neural networks to clone driving behavior. You will train, validate and test a model using Keras. The model will output a steering angle to an autonomous vehicle.

We have provided a simulator where you can steer a car around a track for data collection. You'll use image data and steering angles to train a neural network and then use this model to drive the car autonomously around the track.

We also want you to create a detailed writeup of the project. Check out the [writeup template](https://github.com/udacity/CarND-Behavioral-Cloning-P3/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup. The writeup can be either a markdown file or a pdf document.

To meet specifications, the project will require submitting five files: 
* model.py (script used to create and train the model)
* drive.py (script to drive the car - feel free to modify this file)
* model.h5 (a trained Keras model)
* a report writeup file (either markdown or pdf)
* video.mp4 (a video recording of your vehicle driving autonomously around the track for at least one full lap)

This README file describes how to output the video in the "Details About Files In This Directory" section.

Creating a Great Writeup
---
A great writeup should include the [rubric points](https://review.udacity.com/#!/rubrics/432/view) as well as your description of how you addressed each point.  You should include a detailed description of the code used (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :). 

You're not required to use markdown for your writeup.  If you use another method please just submit a pdf of your writeup.

The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

The following resources can be found in this github repository:
* drive.py
* video.py
* writeup_template.md

The simulator can be downloaded from the classroom. In the classroom, we have also provided sample data that you can optionally use to help train your model.

## Details About Files In This Directory

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.

#### Why create a video

1. It's been noted the simulator might perform differently based on the hardware. So if your model drives succesfully on your machine it might not on another machine (your reviewer). Saving a video is a solid backup in case this happens.
2. You could slightly alter the code in `drive.py` and/or `video.py` to create a video of what your model sees after the image is processed (may be helpful for debugging).

### Tips
- Please keep in mind that training images are loaded in BGR colorspace using cv2 while drive.py load images in RGB to predict the steering angles.

## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).

