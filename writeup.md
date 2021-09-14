# **Traffic Sign Recognition** 

## Writeup

**Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/sample_distribution.png "Visualization"
[image2]: ./examples/color_images.png "Color raining images"
[image3]: ./examples/grayscale_images.png "Color raining images"
[image4]: ./examples/50_sign.jpg "Traffic Sign 1"
[image5]: ./examples/left_turn.jpg "Traffic Sign 2"
[image6]: ./examples/no_entry_sign.jpg "Traffic Sign 3"
[image7]: ./examples/roadworks_sign.jpg "Traffic Sign 4"
[image8]: ./examples/stop_sign.jpg "Traffic Sign 5"

---
### Writeup / README

Link to my: [project code](https://github.com/yosuah/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### 1. Data Set Summary & Exploration

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* Number of training examples = 34799
* Number of validation examples = 34799
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43


#### 2. Exploratory visualization of the dataset.

The following chart shows ratio of samples per category in the train/validation/test sets. 
It can be see that the 3 sets follow a similar distribution, which is what we expect. 
It can also be seen that there are much less samples from some of the categories than the others. 

![alt text][image1]

### 3. Preprocessing

Images are converted to grayscale, because the shapes and labels on the traffic signs already clearly
separate them visually, and color coding is only an auxuliary source of information. Given that
the perceived color can vary a lot depending on factors like the current lighting, it is
expected that removing color information altogether makes it easier for the model to generalize
and avoid overfitting.

Additionally the images are (approximately) normalized using the full range of the possible pixel values (0, 255).

Original color training images (one per label):
![alt text][image2]

Normalized grayscale training images (one per label, different than the color ones):
![alt text][image3]

No image data augmentation was necessary to achieve acceptable performance.

#### Model architecture

I used the LeNet-5 architecture. On top of the model used during the class, I added L2 regularization,
dropout, an option to use max or average pooling and an option to use single-channel or color
input images.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1/3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x64 	|
| RELU					|												|
| Max/avg pooling    	| 2x2 stride, outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, output 10x10x16      									|
| RELU          		|         									|
| Max/avg pooling		| 2x2 stride, valid padding, output 5x5x16        									|
| Flatten				|												|
| Dropout           	| Optional												|
| Fully connected		| 400->120												|
| RELU					|												|
| Dropout               | Optional											|
| Fully connected		| 120->84												|
| RELU					|												|
| Dropout               | Optional											|
| Fully connected		| 84->43												|

 
I collected all hyperparameters and after some manual exploratory work used grid search to find an
optimal combination of parameters for the final solution.
The following hyperparameters were considered:

 * dropout_keep_prob: no dropout, 0.5, 0.7
 * learning rate: 0.01, 0.001, 0.0001
 * weights_sigma: 0.1 (not tweaked)
 * l2_reg_scale (regularization parameter): no regulartization, 0.001, 0.0001 
 * pool_type: max or average
 * epochs: 10-70
 * batch_size: 10-128
 * input_channels: 1 (gray) or 3 (color)

In all cases I used the ADAM optimizer.
For each hyperparameter I saved the maximum and final validation loss (to see if early termination would
help) and used these to select the best performing model.

My final model results were:
* training set accuracy of 0.982
* validation set accuracy of 0.935
* test set accuracy of 0.913


### Test a Model on New Images

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

These images look fairly similar to the original data set, though they might be more agressively cropped 
(meaning that the sign covers almost all of the image, without much backgroud or context).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (50km/h)      	| Keep right   									| 
| Turn left ahead     			| Roundabout mandatory 										|
| No entry					| Stop											|
| Road work	      		| Road work					 				|
| Stop			| Stop      							|


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. 
This is much worse than the results on the test set, suggesting that there are important differences
between these images and the original data set. Despite the errors, the model has somewhat reasonable
output, for example the second image (Turn left ahead), which contains arrows, was misclassified
for another class (Roundabout mandatory) which also contain arrows.

#### Detailed results per new test image

 - INCORRECT: Image of class 2 (Speed limit (50km/h)) classified incorrectly, but correct result is top 5. Top prob: 14.68, correct class probability: 5.84
 - INCORRECT: Image of class 34 (Turn left ahead) classified incorrectly, but correct result is top 5. Top prob: 7.47, correct class probability: 3.13
 - INCORRECT: Image of class 17 (No entry) classified incorrectly, but correct result is top 2. Top prob: 10.95, correct class probability: 4.49
 - CORRECT:   Image of class 25 (Road work) classified correctly. Top prob: 25.02, second probability: 19.57
 - CORRECT:   Image of class 14 (Stop) classified correctly. Top prob: 12.78, second probability: 5.13