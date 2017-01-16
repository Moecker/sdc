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

# Model

# Pre-Processing

# Results
I have taken two video - one per track uploaded on youtube:

* Track 1: https://youtu.be/iqy6XQ0H5s0
* [![https://youtu.be/iqy6XQ0H5s0](https://img.youtube.com/vi/iqy6XQ0H5s0/0.jpg)](https://www.youtube.com/watch?v=iqy6XQ0H5s0)

* Track 2: https://youtu.be/XvbJq8dvNRY
* [![https://youtu.be/XvbJq8dvNRY](https://img.youtube.com/vi/XvbJq8dvNRY/0.jpg)](https://www.youtube.com/watch?v=XvbJq8dvNRY)

Note that the recording are in a quite low quality due to adverse influences of the recording program when recoreded in a higher quality (see also 'Other' for details)

# Other
To document the performance of the trained network on both tracks I atempted to screen-record a video of the Simulator's Autonomous Mode. It was noticed that the actual recording influenced the notebooks performance adversly so that the Simulator struggled to output enough frames per second - or at least much less than without the recording. 
This eventually lead to a worse performance of the autonomous drive since the network was expecting a higher frames per second rate. In detail what happend was that the car was osciliating within the lane, which did not happen without recording. 
I would not blame the recording in particular, but would in general say that the network and the Simulator is not robust for a variation in frames per seconds.
