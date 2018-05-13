# **Traffic Sign Recognition**

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/original_graph.png "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples_from_internet/2.jpg "Traffic Sign 1"
[image5]: ./examples_from_internet/3.jpg "Traffic Sign 2"
[image6]: ./examples_from_internet/4.jpg "Traffic Sign 3"
[image7]: ./examples_from_internet/5.jpg "Traffic Sign 4"
[image8]: ./examples_from_internet/6.jpg "Traffic Sign 5"
[image9]: ./examples/predicted.png "Predicted"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed amongst different traffic sign classes. There seems to be a excess amount of particular sign images and very low of others. This can potentially cause our neural network to be bias towards those traffic signs. If I had enough time I would create more fake data by applying image manipulation to create more data for signs with low amounts of data. I would use techniques such as image rotation, grayscale, warp and etc.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it simplifies learning. Leaving it as a color image would only cause learning time to increase because more data is relayed. By grayscaling we only give data that is crucial to learn such as sign shape and size.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data to have slight rotation. They still look alike but have slight variations.

If given more time I would have generated more training data. I tis important to have the same amount of average data per class to avoid bias.



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 28x28x10	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x10 				|
| Convolution 5x5	    | 2x2 stride, valid padding, outputs 10x10x20			|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x20 				|
| Convolution 5x5	    | 2x2 stride, valid padding, outputs 1x1x500			|
| RELU					|												|
| Flatten		| input 5x5x20, output 500       									|
| Fully connected		| input 500, output 120       									|
| RELU					|												|
| Fully connected		| input 120, output 84       									|
| RELU					|												|
| Fully connected		| input 84, output 10     									|




#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used a LeNet model and trained it on the German Sign dataset. I used a learning rate of 0.00005. I tried using 0.0001 as a learning set but the training would take a lot of time. After trying different amount of epochs, I used 100 as it delivered training accuracy to be more than 99% and the validation accuracy to be more than 94%. I also choose a batch size of 500 to increase the training speed.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.992
* validation set accuracy of 0.947
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?     
    It was implemented in the same way because the research that was done used this architecture.
* What were some problems with the initial architecture?
    Mainly the first architecture i choose had a low epochs and low batch size. I tweaked those parameters to optimize the solution.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

    As mentioned in the previous answers, I didn't add more convolution layers to the architecture but rather tweaked the batch size, learning rate and epochs parameters.
* Which parameters were tuned? How were they adjusted and why?
   Batch size, learning rate and epochs parameters were tweaked to optimize training time and accuracy.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
   Convolution layers work well for this problem because it isn't gauranteed that the images are always going to be centered but rather different in many cases. By applying convolution to the image we can generalize the image more but not having the area of the sign in an given image be important.
If a well known architecture was chosen:
* What architecture was chosen?
   LeNet
* Why did you believe it would be relevant to the traffic sign application?
   Because it was built for recognizing letters from an image, and recognizing traffic signs isn't any different.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
   The model training, validation and test accuracy are over 90% which prove it is the write architecture for this problem.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because it kind of has an arrow pointing right and left which outputs as turn right ahead instead of no entry.

![alt text][image9]

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| No Stopping     		| Turn right ahead 									|
| No Entry    			| No entry 										|
| General caution				| General Caution											|
| Yield	      		| Yield					 				|
| 30 km/h			| 30 km/h     							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 99%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 36th cell of the Ipython notebook.

For the two images the probability was the same at 21%. But of the third one it went down because it wasn't sure about the one i got wrong. The fifth was the highest probability as it had numbers and was the most different.

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .21         			| Turn right ahead   									|
| .21     				| No entry 											|
| .15					| General Caution										|
| .18	      			| Yield				 				|
| .26				    | 30 km/h     							|



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
