# Violence Audio Identification using Tensorflow
###### A part of Zupa App by Bangkit 2021 CAP0175 capstone project.

## Main purpose
To identify four types of violence (Sexual harrastment/violence, domestic violence, stalking, and physical threat/violence) from an audio recording.

## Brief Overview

![](img/overview.png)

As seen above, we took the MFCC values of the audio recording, and then feed them into Convolutional Neural Network (CNN). The CNN output is a softmax layer that will return four float values in two-dimentional array. Those values represents probability for each violence category.

## Dataset
We collected our own dataset, and we will publish the dataset. Details and release of our dataset can be found [here](https://www.google.com). We asked our voluntee to record 3 to 5 seconds audio as if they were in 40 different situations delivered in short scripts. Each situation categorised manually. One audio may fall into two categories. For example, some of sexual violence followed by a physical violence beforehand. So when we determine the occurence of both abuse in single audio file, we duplicate the audio and put them into corresponding folders. We realised that our data is too small compared to the usual dataset for neural network implementation. 

## Convolutional Neural Network
We found that our model was highly overfitted. This CNN model initialy used for [TensorFlow Speech Recognition Challenge](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge). Here is our best implementation for this problem:

```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 42, 11, 64)        640       
_________________________________________________________________
batch_normalization (BatchNo (None, 42, 11, 64)        256       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 21, 6, 64)         0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 19, 4, 32)         18464     
_________________________________________________________________
batch_normalization_1 (Batch (None, 19, 4, 32)         128       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 10, 2, 32)         0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 9, 1, 32)          4128      
_________________________________________________________________
batch_normalization_2 (Batch (None, 9, 1, 32)          128       
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 5, 1, 32)          0         
_________________________________________________________________
flatten (Flatten)            (None, 160)               0         
_________________________________________________________________
dense (Dense)                (None, 64)                10304     
_________________________________________________________________
dense_1 (Dense)              (None, 4)                 260       
=================================================================
Total params: 34,308
Trainable params: 34,052
Non-trainable params: 256
_________________________________________________________________

```

## Files and Folders
- `/'dataset` is where we store our dataset.
- `/tflite` is where our model is.
- any `.json` file is our audio features used for training.
- Python Notebook (`.ipynb`) files for straight python implementation. We made them seperate. But if you want to see it prepare, train, and test the model in one go, use `prepare-train-predict.ipynb`
- any `.h5` file is a saved neural network model.
- `/img` is only for this README purpose.


## Deployment
As this model will be deployed in **Zupa Mobile App**, we convert the saved H5 model into Tensorflow Lite model. We tried serveral configuration labelled with number. You can find the H5 and Tensorflow Lite model in `/tflite` folder. Please use the H5 model for regular use and `model.tflite` in each folder purposed for mobile implementation of this model.

## Contribute! (we will be super happy!)
In advance we want to thank you for any kind of your support! Your support drives us to enhace our application and reach our goal in women, children, and public safety. If you want to contribute especially in this machine learning model, kindly purpose a pull request. If you want to contribute to the dataset, please visit the dataset dedicated [github page]() to see our notes, scripts, etc. Then contact our machine learning engineer [with email](mailto:iga.narendra@gmail.com). Don't hesitate to tell us your story. You can write your story inside the email or have a safe, under COVID-19 protocol meet (can be online or onsite around Jakarta, Bandung and Denpasar).