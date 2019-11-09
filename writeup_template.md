# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road

[//]: # (Image References)

[net]: ./examples/model.png "Model Visualization"
[original]: ./examples/original.png "Model Visualization"
[cropped]: ./examples/cropped.png "Model Visualization"
[brightness]: ./examples/random-brightness.png "Model Visualization"
[flipped]: ./examples/flipped.png "Model Visualization"

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

---

#### 1. Files

My project includes the following files:
* model.py: contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.
* drive.py: for driving the car in autonomous mode
* preprocessing.py: for pre-processing video frames
* model.h5: containing a trained convolution neural network 
* writeup.md: summarizing the results

#### 2. Functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

### Model Architecture and Training Strategy

#### Model architecture

As suggested in the Udacity lecture I implemented the NVIDIA CNN model. This CNN was presented in [End to End Learning for Self-Driving Cars](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
and is known to be very successful for this task. The table below shows my implementation of this CNN in Keras:

![alt text][net]

The input of this network is an image with a shape of (33, 66, 200). The network itself consists of three 5x5 convulution layers, two 3x3 convulution layers, and three fully converted layers.
As the activation function, I started with `relu`, but later I changed to `elu`. The `elu` activation function has the advantage that it does not have the dying ReLU problem. 

Moreover, I played around with Dropout and MaxPooling layers. Unfortunately, the model performance on the track did get worse with these layers. So I excluded them from the network. Instead I used L2 kernel regularizers to reduce overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

The model used an adam optimizer and a mean-squared error loss function.

#### Training Process and Creation of Training Data

The image below shows an example input image to the process pipeline:

![alt text][original]

For the training process I had to pre-process this images. First of all, unimportant sections of the image are removed:

![alt text][cropped]

Then the image is converted to YUV colorspace and normalised: `img = img/127.5 - 1`. Finally, the image is resized to a size of (33, 66, 200). This is the suggested image shape for the NVIDIA CNN.

In order to increase the diversity, data augmentation is used. Therefore the brightness of the images is randomly adjusted:

![alt text][brightness]

Besides a lot of straight sections the test track mainly consists of left turn. Thus, the image is flipped in order to artificially create right turns:

![alt text][flipped]

I started to train my model with the data set provided by Udacity. The trained model performed quite well on the test track, but struggled to stay on the lane at some poins.
Thus, I decided to create my own data set. I 

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Furthermore, I recorded scenes that show what oen has to do when the car gets close to the border.
This scenes are essential since without these scenes the model could not learn what do in these situations.

In training mode, the simulator produces three images per frame while recording corresponding to left-, right-, and center-mounted cameras, each giving a different perspective of the track ahead. 
I used all of these images. 



After the collection process, I had 6301 number of data points. This data set had a lot of straight sections. Thus I randomly removed 75% of all images in straight sections.
Thereby, the data set is reduced to 2356 image frames.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting.

The following parameters where used during the training process:

* Epochs : 10
* Batch Size: 32
* Regularization Rate: 0.001

Due to the small number of data samples, there was no need to use a GPU. The training process was also fast enough with just a CPU.

The trained model was able to autonomously drive around the test track for several laps. Moreover, it mostly performed quite natural and did not shake around much.