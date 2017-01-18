# Introduction
This GitHub Repository contains the code targeting the Behavioral Cloning Project of the Udacity Self Driving Car Nanodegree.
The Behavioral Cloning Project aims to copy the human control of a car in a simulator. It uses a Convolutional Deep Neural Network to train a model based on virtual camera images taken from the simulator's car.

It hence shall mimik a real conceivable training task of a real car taken images from a real road. 

The task was to:
* first, record training data in simulator's training mode by driving the track yourself. The actual recording of images was already built in the simulator. Alternatively it was possible to use training images provided by Udactiy. I chose the latter option.
* second, train a network using the recorded images and a csv file containing the corresponding steering angles per image. The networks layout was choosen free of choice.
* third, evaluate the autonomous driving performance in the simulator's autonomous mode. The simulator hereby takes the trained model and it weights to instantenly compute the correct steering angle for each frame.

# Approach
The approach for the project was a mix between reading trough the mentioned NVidia paper, helpful Medium blog posts and Slack message, recommended articels by other SDC-lers and a handful of engineering try-and-error. 

## Recovery
We can incorporate the left and right camera images to simulate recovery, by adding or subtracting an artificial steering angle to the center steering value according to the direction.

# Data Description
As described in te exploration document [exploration.html](exploration.html), the data set consists of mainly the two inputs:
* driving_log.csv
* images of three virtual cmaeras

The *driving_log.csv* contains logs of a training session, each recorded frame per row, structured as follow:

|center|left|right|steering|throttle|brake|speed|
|---|---|---|---|---|---|---|
|IMG/center_...jpg|IMG/left_...jpg|IMG/right_....jpg|0.1765823| 0.9855326|0|30.18422|

The *IMG* folder contains the actual captured images in jpg format, visualized as follow:

![cameras](exploration/2017-01-18 21_41_42-exploration.png)

## Distribution
By plotting the distribution of the steering angles, we can observe that most of the time the steering angle is close to zero
* Strong steering angles are very rare
* There is a bias towards steering angles which are positive
Hence the data set is very unbalanced towards small steering angles which can be problematic in strong curves. Also the bias of positive angles can be problematic.

![distribution_before](exploration/2017-01-18 21_42_48-exploration.png)

## Countermeasures
* Countermeasuresfor positive steering angle bias: The images can be flipped horizontally (and invert the corresponding steering angle), so that we can reduce the bias for turing left (see section 'Pre-Processing')
* Countermeasure for small steering angle bias: Possible solution is to increase the number if images and steering angles where we detect a high degree of curvature, based on the actual input steering angle. By duplicating those detected images we get a slighty more balanced dataset.

![distribution_after](exploration/2017-01-18 21_43_00-exploration.png)

# Model
The model of the network used for training is a sequential keras model with five layers.

1. The first layer is a 2D Convolution with an ELU activation, using the images input shape (kRows = 64, kCols = 64, kChannels = 3) and outputs a shape of 32x32x32. The convolution uses a subsample of (2, 2) to reduce the number of pixels and same padding, which in total reduced the number of pixel dimension by half.

2. In the second layer another Convolution layer is applied, followed by a MaxPooling stride, activated by an ELU. The first Dropout layer with a keep probablity of 40% is introduced to prevent overfitting. The output shape is 15x15x16.

3. A last Convolution layer with again ELU activation, Droput with same 40% keep probability is followed. The Output shape is 13x13x8, since the subsample is (1, 1) with valid padding.

4. A flatten layer transforms the 3 dimensional (4 if you take the actual batches into account) into a flat one.

5. The fourth layer is the first Dense one, reducing the dimension to 1024, using Dropout and likewise an ELU activation.

6. Eventually a Dense layer with ELU activastion closes the main layer-stack reducing the output shape to 512.

7. Since we are not interested into a classification but rather are confornted with a regression problem, we add a simple single dense layer of output shape, which can be understood as the actual value of the output steering angle.

## Model Visualized
The model is summarized and visualized as follow:

![model](model.png)

# Pre-Processing

First, each image was cropped at the top and bottom, rationale:
* The actual visible road is superimposed by the vehicle's body
* The image contains a lot of sky which is not helpful to train the network

|The raw input image|The cropped image|
|---|---|
|![normal](exploration/normal.png)|![cropped](exploration/cropped.png)|

## Augmentation
### Use left & right camera images to simulate recovery
Using left and right camera images to simulate the effect of car wandering off to the side, and recovering. We will add a small angle 0.25 to the left camera and subtract a small angle of 0.25 from the right camera. The main idea being the left camera has to move right to get to center, and right camera has to move left.
#### Flip the images horizontally
Since the dataset has a lot more images with the car turning left than right(because there are more left turns in the track), you can flip the image horizontally to simulate turing right and also reverse the corressponding steering angle.
#### Brightness Adjustment
In this you adjust the brightness of the image to simulate driving in different lighting conditions
#### Add Random Shadows
By adding random shadows we can increase the number of augmented images to as many as we want. The idea behind the shadowing is that the network will be trained to detect the actual important edges of the street and no edges introduced by bad lightning conditons.

# Results
I have taken two video - one per track uploaded on youtube:

* Track 1: https://youtu.be/iqy6XQ0H5s0

[![https://youtu.be/iqy6XQ0H5s0](https://img.youtube.com/vi/iqy6XQ0H5s0/0.jpg)](https://www.youtube.com/watch?v=iqy6XQ0H5s0)

* Track 2: https://youtu.be/XvbJq8dvNRY

[![https://youtu.be/XvbJq8dvNRY](https://img.youtube.com/vi/XvbJq8dvNRY/0.jpg)](https://www.youtube.com/watch?v=XvbJq8dvNRY)

Note that the recording are in a quite low quality due to adverse influences of the recording program when recoreded in a higher quality (see also 'Other' for details)

# Acknowledgment
To document the performance of the trained network on both tracks I atempted to screen-record a video of the Simulator's Autonomous Mode. It was noticed that the actual recording influenced the notebooks performance adversly so that the Simulator struggled to output enough frames per second - or at least much less than without the recording. 
This eventually lead to a worse performance of the autonomous drive since the network was expecting a higher frames per second rate. In detail what happend was that the car was osciliating within the lane, which did not happen without recording. 
I would not blame the recording in particular, but would in general say that the network and the Simulator is not robust for a variation in frames per seconds.

# Credits
Credits shall go in particular to the following blog post, papers and articles:
* http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
* https://medium.com/@subodh.malgonde/teaching-a-car-to-mimic-your-driving-behaviour-c1f0ae543686#.r9rvmm3so
* https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.itxajj34m
* https://carnd-forums.udacity.com/cq/viewquestion.action?id=26214464&questionTitle=behavioral-cloning-cheatsheet
* https://carnd-udacity.atlassian.net/wiki/pages/viewpage.action?pageId=30441475
