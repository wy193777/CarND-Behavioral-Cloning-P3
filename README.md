#**Behavioral Cloning**

[//]: # (links)

[nvidia]: https://arxiv.org/abs/1604.07316 "End to End Learning for Self-Driving Cars"


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* Train Data.ipynb containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model2.h5 containing a trained convolution neural network
* README.md summarizing the results

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is come from [NVIDIA's model][nvidia], but removed several layers. The second and forth convolution + RELU layers are being removed because I guess a simulated road is much simpler than real roads.

Input images are first been cropped and then normalized.


#### 2. No over-fitting find.

I guess because I use more data and less layers and didn't trained too much, I couldn't noticed an over-fitting problem.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road from both tracks.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I simply follow instruction videos and [NVIDIA's paper][nvidia]. Using CNN makes so much sense for me because the data comes from front cameras and the car need to follow features on the road like human beings.

Except using all center, left and right camera images, I also mirrored the image to prevent left turning tendency.

I commented out several layers to make the model simpler.

I first trained the model using Udacity's data. The car works fine on first track but go out of the road at the very beginning of the second track.

Then I recorded the data on the second track. Ideally the car should always keep on right lane but I don't have enough patient to practice. So even though I'm a bad driver, I successfuly keep the car on the road in most conditions.

Then I combined Udacity's data and my drive data from second track together and feed them into the same network. The result is good. The car could drive on both tracks.

#### 2. Final Model Architecture

The final model architecture (Train Data.ipynb Train Model section) consisted of a convolution neural network with the following layers:

 * Cropping image's top 70 rows and bottom 25 rows. 160x320x3 to 65x320x3
 * Normalize the image
 * Convolution layer 5x5 kernel, output 24@30x158 with relu.
 * Convolution layer 5x5 kernel, output 48@13x77 with relu.
 * Convolution layer 3x3 kernel, output 64@4x35 with relu.
 * Flatten to 8960 neurons.
 * Fully connected 100 neurons.
 * Fully connected 10 neurons.
 * Fully connected 1 neurons to output steer angles.

#### 3. Creation of the Training Set & Training Process

Combined Udacity's data and mine data from second track.

After the collection process, I had 13364 data points.


I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was close to human behaviors. I used an adam optimizer so that manually training the learning rate wasn't necessary.
