#**Traffic Sign Recognition** 
**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/christianreiser/P2-Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier%20v1.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The size of validation set is 4410
* The shape of a traffic sign image is 32x32 with 3 color channels
* The number of unique classes/labels in the data set is 43


####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the 3. until 5. code cells of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how some classes have lots of images while some classes just have a few.

![](https://github.com/christianreiser/P2-Traffic-Sign-Classifier/blob/master/Images/classes2.png)
Here is the first image of each class:
![alt text](https://github.com/christianreiser/P2-Traffic-Sign-Classifier/blob/master/Images/classes.png)
Some signs are even for a human difficult to see.
![alt text](https://github.com/christianreiser/P2-Traffic-Sign-Classifier/blob/master/Images/9random.png)

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the 6. til 8. code cells of the IPython notebook.

I normalized the image data because its easier for the CNN to train with normalized data.
Normalized image:
![alt text](https://github.com/christianreiser/P2-Traffic-Sign-Classifier/blob/master/Images/normal.png)

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)
 

The data set was already split into a validation, training and testset.

My final training set had 34799 number of images. My validation set and test set had 4410 and 12630 number of images.


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the 9. cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 28x28x18 	|
| RELU					|												|
| Dropout					|		0.75										|
| Max pooling	      	| 2x2 stride,  outputs 14x14x18. 	VALID padding			|
| Convolution 5x5	    |  1x1 stride, VALID padding, outputs 10x10x48 									|
| RELU					|												|
| Dropout					|		0.75										|
| Max pooling	      	| 2x2 stride,  outputs 5x5x48. 	VALID padding			|
|	Flatten					|	outputs 1200											|
| Fully connected		| outputs 300        									|
| RELU					|												|
| Dropout					|		0.75										|       									|
| Fully connected		| outputs 100        									|
| RELU					|												|
| Dropout					|		0.75										|
| Fully connected		| outputs 43        									|
| Softmax				| to get probabilities |

 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the 10. cell of the ipython notebook. 

To train the model, first used a batch size of 128. When optimizing my architecture I wasn't sure if thats a good size. I read in the internet, that a bigger batch size leads do a more acurate gradient, but my memory could be to small. Consequently, I increased my batch site and watched my memory. I found that I could increase the match size to more than 3000. However, the valiation accuracy droped. Later I read that when the batch size is too big, the gradient could be very accurate on the test images, but is not able to generalize as well. So I returned to a batch size of 128 again and also tried to lower it even more to 100. A size of 100 seemed to give the best results. However, lowering it even more could still be beneficial.
At first I set the learning rate to 0.00001 with 200 epoches. But it was too time consuming and I couldn't iterate on my odel fast enough. So I lowerd the epoches to 15 and set the learning rate to 0.001. In the last 5 epoches it seems like the Network is not learning anymore, possibly 10 or 11 Epoches would work too. From the beginning I used the Adam Optimizer.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the 11. cell of the Ipython notebook.

My final model results were:
* validation set accuracy of 93.6% 
* test set accuracy of 92.7%
* My first architecture was very similar to the LeNet architecture.
* problems with the initial architecture were, that LeNet was optimized for gray images, but I wanted to feed RGB images
* I added dropout and also mistakenly reduced the learning rate too much.
* I adjusted the dropout rate because I tought I was overfitting.
* I think the CNN was a great choice, because the images are not always in the same place.


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.
These are my five German traffic signs from the web:


####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).
There are five German traffic signs that I found on the web

![alt text](https://github.com/christianreiser/P2-Traffic-Sign-Classifier/blob/master/Images/5int.png)

I also normalized the images:

![alt text](https://github.com/christianreiser/P2-Traffic-Sign-Classifier/blob/master/Images/norm5int.png)

The second and third image might be difficult to classify because they are both similar looking speed limits.
The model was not able to detect the new Stopp sign and I'm not sure why it is wrong with a certainty of 100%.
The other images were easy to classify.


The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	       		| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| priority road->100% 		| 
| speed limit 30     			| speed limit 30->57%;speedlimit 50km/h with 40% |
| speed limit 50 					| speed limit 50->49%; speedlimit 30km/h with 44% |
| roundabout mandatory      		| roundabout mandatory -> 100%  	|
| Right of way at next intersection		| Right of way at next intersection -> 100%  |


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. Thats less as in the test. Maybe because the images are uncut
1. Right of way at next intersection: correct classification with a certainty of 100%
2. Speedimit 50km/h: correct classification with a certainty of 49%, next guesses are speedlimit 30km/h with 44% and speedlimit 80km/h with 7%
3. Speedimit 30km/h: correct classification with a certainty of 57%, next guesses are speedlimit 50km/h with 40% and speedlimit 50km/h with 2%
4. Roundabout mandatory: correct classification with a certainty of 100%
5. Stop: INCORRECT classification with a certainty of 100%. The stop sign is classified as a priority road.

Interpretation:
For the model Right of way at next intersection and Roundabout mandatory are very easy to distinguish.
The the model is less certain about the speedlimits, nevertheless it classifies correctly.
The incorrect prediction with a wrong certainty of 100% is very strange. At first I tought my labeling could be wrong. However I can't finde the mistake. Maybe its due to the uncut image in contrast to the cut training images.
