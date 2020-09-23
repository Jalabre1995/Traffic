# Traffic
Creating a AI that uses Neural Networks to tell whether or not the image is a Traffic light.

The files that are in this directory is  German Traffic Sign Recognition Benchmark (GTSRB) dataset, which contains thousands of images of 43 different kinds of road signs. A requirements.txt file, and a traffic.py file. 

# Dependencies
````
pip3 install -r requirements.txt
import scikit-learn and Tensorflow for Machine Learning
import pandas as pd
import matplotlib.pyplot
import cv2 
````

# Results
When getting the results, I wanted to play around with the data and see if changing some of the properties in the neural network would change the accuracy of the results. First I wanted to know the difference between using the softmax activation model or the sigmoid activation model. I tried the sigmoid model first while using a binary_crossentropy. And for the sigmoid, binary the results were around 99% accuracy, but for the sigmoid categorical, the highest accuracy was 96%. And for softmax, the binary was around 96% and 95% for categorical. The reason is that the sigmoid is making outputs between 0 and 1. So i.e, Sigmoid is useful when you are comparing two objects. That is why when you are using Sigmoid, you also want to use binary_crossentropy as well, due to comparing two values. For this particular problem, categorical and softmax would be better to use, because softmax ensures that the sum of outputs along channels (as per specified dimension) is 1 i.e., they are probabilities. Out of 43 categories, the AI is getting a more accurate percentage due to all the inputs given. 

What I wanted to test next is the pool_size. For the pool_size, the data is able to return the categories 10 times with a 2,2 pooling_size.
I changed the pooling_size for 32, 64, and 128 to 3x3 kernal(3,3). And the results loaded all 26640 from 43 categories, but an error kept showing up on the terminal. The reason could be that 3x3 kernal is not even for the filters. Meaning that an input won't be counted for. I also tried 4x4 kernal and see what would happen and recived an valueError similar to the input of the 3x3 kernal.

Next was playing around with the dropout rate. For the dropout rate, I increased the rate to 0.7 and an error came up for the first epoch saying:
```
WARNING:tensorflow:Large dropout rate: 0.7 (>0.5). In TensorFlow 2.x, dropout() uses dropout rate instead of keep_prob. Please ensure that this is intended.
```
And the accuracy is around 85%. When I did it at 0.5, the accurcy was still around 94%. This is due to a lot of nodes being dropped out when the data is trained. In conclusion for the dropout rate, having the rate around at 0.4 and 0.5 is a good amount of data to be trained and tested. 
