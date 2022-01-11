# Convolutional Neural Network
### Problem statement:

Handwritten Digit Classification using MNIST dataset.

MNIST is a dataset of 60,000 training set images of handwritten single digits between 0 and 9,
each image is a 28x28 pixel square.


The task is to classify a given image of a handwritten digit into on of 10 classes representing 
integer values from 0 to 9, inclusively.


- Do Preprocessing step (Normalization). Rescale pixel values to the range 0-1
- Build 4 different architecture CNN models that can detect the digit of a given image 
 (change number of convolutional layers, pooling layers, etc)
- Apply cross validation during training, and shuffle training dataset at each epoch.
- Evaluate the models using accuracy
