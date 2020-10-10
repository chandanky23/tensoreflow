import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

# datasets with hand written letters
mnist = keras.datasets.mnist

# get the test and train data from mnist load_data() method
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# (60000, 28, 28) (60000,)
# -> we have 60000 training images and each image has 28 X 28 size.
# -> we have 60000 labels
print(x_train.shape, y_train.shape)

# Normalize the data as the images has values between 0, 255 -> to 0,1
x_train, x_test = x_train/255.0, x_test/255.0

# lets plot the data (for first 6 images)
for i in range(6):
  plt.subplot(2,3, i+1)
  plt.imshow(x_train[i], cmap='gray')
# plt.show() -> uncomment to check the 6 handwritten images

# model
model = keras.models.Sequential([
  keras.layers.Flatten(input_shape=(28,28)), # Reduce one dimension of each images (28 X 28) -> flatten our images
  keras.layers.Dense(128, activation='relu'), # The dense layer is the fully connected layer in keras api. We can also use something different than 128
  keras.layers.Dense(10) , # 10 here reflects number of outputs (i.e, number of classes to classify). should always be equal to classes.
])
# ReLU :
# Known as "rectified linear activation function".
# It is a linear function that will output the input directly if its positive, otherwise zero.
# It is now considered as a default activation function as its easy to train with and performs better.
# It overcomes the vanishing gradient problem thus making models perform better.

print(model.summary()) # this will show the below result if we pass `input_shape=(28,28)` to Flatten as parameter above.

"""Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
flatten (Flatten)            (None, 784)               0         
_________________________________________________________________
dense (Dense)                (None, 128)               100480    
_________________________________________________________________
dense_1 (Dense)              (None, 10)                1290      
=================================================================
Total params: 101,770
Trainable params: 101,770
Non-trainable params: 0
_________________________________________________________________
None
"""

# The above model can also be created in the following fashion. The advantage is, we can then print the summary after each.

# model = keras.Sequential()
# model.add(keras.layers.Flatten(input_shape=(29,28)))
# print(model.summary())
"""
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
flatten (Flatten)            (None, 812)               0         
=================================================================
Total params: 0
Trainable params: 0
Non-trainable params: 0
"""

# model.add(keras.layers.Dense(128, activation='relu'))
# print(model.summary())
"""
_________________________________________________________________
None
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
flatten (Flatten)            (None, 812)               0         
_________________________________________________________________
dense (Dense)                (None, 128)               104064    
=================================================================
Total params: 104,064
Trainable params: 104,064
Non-trainable params: 0
"""

# model.add(keras.layers.Dense(10))
# print(model.summary())
"""
_________________________________________________________________
None
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
flatten (Flatten)            (None, 812)               0         
_________________________________________________________________
dense (Dense)                (None, 128)               104064    
_________________________________________________________________
dense_1 (Dense)              (None, 10)                1290      
=================================================================
Total params: 105,354
Trainable params: 105,354
Non-trainable params: 0
"""

# Loss and optimizer
# this is because our Y is an integer class label (0,1,2,3..)
# from_logits=True as we didn't include the softmax in the above model after Dense layer.
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
"""
for y = 0, y=[1,0,0,0,0,0,0,0,0,0]
then we use loss = keras.losses.CategoricalCrossentropy
"""

# -> lr=0.001, also called "learning rate" is one of the most importamt hyper-parameters.
optim = keras.optimizers.Adam(lr=0.001)
# metrics we need to track
metrics = ["accuracy"]

model.compile(loss=loss, optimizer=optim, metrics=metrics)

# training
# The batch size is the amount of samples you feed in your network at a particluar time out of training samples.
batch_size = 64
# An epoch is a measure of the number of times all of the training vectors are used once to update the weights.
epochs = 5

# verbose=2 -> normal logging,
# shuffle=True is by default
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=2)
"""
Epoch 1/5
938/938 - 1s - loss: 0.2988 - accuracy: 0.9161
Epoch 2/5
938/938 - 1s - loss: 0.1347 - accuracy: 0.9606
Epoch 3/5
938/938 - 1s - loss: 0.0939 - accuracy: 0.9726
Epoch 4/5
938/938 - 1s - loss: 0.0712 - accuracy: 0.9788
Epoch 5/5
938/938 - 1s - loss: 0.0566 - accuracy: 0.9832
"""
# it looks good as we have very less loss

# evaluate
model.evaluate(x_test, y_test, batch_size=batch_size, verbose=2)
"""
157/157 - 0s - loss: 0.0821 - accuracy: 0.9744
"""

# Predictions (need softmax to evaluate probability)
probability_model = keras.models.Sequential([
  model,
  keras.layers.Softmax()
])

predictions = probability_model(x_test)
ist_pred = predictions[0] # ist predicted data
print(ist_pred)
"""
tf.Tensor(
[1.3906885e-06 1.5639858e-09 3.3765996e-04 4.8198844e-03 1.2539546e-08
 2.2541080e-07 1.7201671e-10 9.9481732e-01 1.0524035e-05 1.2976390e-05], shape=(10,), dtype=float32)
"""

# Choose the class with highest proabablity
ist_label = np.argmax(ist_pred)
print(ist_label)
""" 7 -> 9.9481732e-01 """

# Another way to do it is to use our model and softmax seperately
predictions = model(x_test)
predictions = tf.nn.softmax(predictions)
print(predictions[0])
"""
tf.Tensor(
[4.1372806e-07 2.4607424e-08 8.1214966e-05 1.0288425e-03 7.0613787e-10
 1.0503344e-06 6.7621221e-12 9.9887747e-01 5.2149940e-06 5.7505272e-06], shape=(10,), dtype=float32)
"""
ist_label = np.argmax(ist_pred)
print(ist_label)
""" 7 -> 9.9481732e-01 """

# Lets get predictions for first 5 samples in test data
ist_5_prediction = predictions[0:5]
print(ist_5_prediction.shape)
""" (5, 10) """
ist_5_label = np.argmax(ist_5_prediction, axis=1)
print(ist_5_label)
"""[7 2 1 0 4]""" # ist prediction is for number 7, second for 2 and so on....

# plotting the ist 5 images
for i in range(5):
  plt.subplot(2,3, i+1)
  plt.imshow(x_test[i], cmap='gray')
plt.show()