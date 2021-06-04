# Violence Audio Identification using Tensorflow
###### A part of Zupa App by Bangkit 2021 CAP0175 capstone project.

## Main purpose
To identify four types of violence (Sexual harrastment/violence, domestic violence, stalking, and physical threat/violence) from an audio recording.

## Brief Overview

![](img/overview.png)

As seen above, we took the MFCC values of the audio recording, and then feed them into Convolutional Neural Network (CNN). The CNN output is a softmax layer that will return four float values in two-dimentional array. Those values represents probability for each violence category.

