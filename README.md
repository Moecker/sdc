# Welcome
Behavioral Cloning Project of the Udacity Self Driving Car Nanodegree

# Introduction
This GitHub Repository contains the code targeting the Behavioral Cloning Project of the Udacity Self Driving Car Nanodegree.
The Behavioral Cloning Project aims to copy the human control of a car in a simulator. It uses a Convolutional Deep Neural Network to train a model based on virtual camera images taken from the simulator's car.

It hence shall mimik a real conceivable training task of a real car taken images from a real road. 

The task was to:
* first, record training data in simulator's training mode by driving the track yourself. The actual recording of images was already built in the simulator. Alternatively it was possible to use training images provided by Udactiy. I chose the latter option.
* second, train a network using the recorded images and a csv file containing the corresponding steering angles per image. The networks layout was choosen free of choice.
* third, evaluate the autonomous driving performance in the simulator's autonomous mode. The simulator hereby takes the trained model and it weights to instantenly compute the correct steering angle for each frame.

# Data Desciption
As described in te exploration document [exploration.html](exploration.html), the data set consists of mainly the two inputs:
* driving_log.csv
* images of three virtual cmaeras

The *driving_log.csv* contains logs of a training session, each recorded frame per row, structured as follow:

|center|left|right|steering|throttle|brake|speed|
|---|---|---|---|---|---|---|
|IMG/center_...jpg|IMG/left_...jpg|IMG/right_....jpg|0.1765823| 0.9855326|0|30.18422|

The *IMG* folder contains the actual captured images in jpg format, visualized as follow:

![cameras](exploration/2017-01-18 21_41_42-exploration.png)

# Model
The model of the network used for training is a sequential keras model with five layers.

1. The first layer is a 2D Convolution with an ELU activation, using the images input shape (kRows = 64, kCols = 64, kChannels = 3) and outputs a shape of 32x32x32. The convolution uses a subsample of (2, 2) to reduce the number of pixels and same padding, which in total reduced the number of pixel dimension by half.

2. In the second layer another Convolution layer is applied, followed by a MaxPooling stride, activated by an ELU. The first Dropout layer with a keep probablity of 40% is introduced to prevent overfitting. The output shape is 15x15x16.

3. A last Convolution layer with again ELU activation, Droput with same 40% keep probability is followed. The Output shape is 13x13x8, since the subsample is (1, 1) with valid padding.

* A flatten layer transforms the 3 dimensional (4 if you take the actual batches into account) into a flat one.

4. The foruth layer is the first Dense one, reducing the dimension to 1024, using Dropout and likewise an ELU activation.

5. Eventually a Dense layer with ELU activastion closes the main layer-stack reducing the output shape to 512.

* Since we are not interested into a classification but rather are confornted with a regression problem, we add a simple single dense layer of output shape, which can be understood as the actual value of the output steering angle.

The model is summarized and visualized as follow:

![cameras](model.png)

# Pre-Processing

# Results
I have taken two video - one per track uploaded on youtube:

* Track 1: https://youtu.be/iqy6XQ0H5s0

[![https://youtu.be/iqy6XQ0H5s0](https://img.youtube.com/vi/iqy6XQ0H5s0/0.jpg)](https://www.youtube.com/watch?v=iqy6XQ0H5s0)

* Track 2: https://youtu.be/XvbJq8dvNRY

[![https://youtu.be/XvbJq8dvNRY](https://img.youtube.com/vi/XvbJq8dvNRY/0.jpg)](https://www.youtube.com/watch?v=XvbJq8dvNRY)

Note that the recording are in a quite low quality due to adverse influences of the recording program when recoreded in a higher quality (see also 'Other' for details)

# Other
To document the performance of the trained network on both tracks I atempted to screen-record a video of the Simulator's Autonomous Mode. It was noticed that the actual recording influenced the notebooks performance adversly so that the Simulator struggled to output enough frames per second - or at least much less than without the recording. 
This eventually lead to a worse performance of the autonomous drive since the network was expecting a higher frames per second rate. In detail what happend was that the car was osciliating within the lane, which did not happen without recording. 
I would not blame the recording in particular, but would in general say that the network and the Simulator is not robust for a variation in frames per seconds.
