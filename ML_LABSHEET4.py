#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Abhishek R S-EEE20005
import tensorflow as tf
import matplotlib.pyplot as plt
# imports the TensorFlow library for machine learning tasks and the matplotlib.pyplot module for data visualization.

mnist = tf.keras.datasets.mnist#assigns the MNIST dataset to the variable mnist.
# The MNIST dataset is a widely used dataset in machine learning 
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# loads the MNIST dataset using the load_data() function from mnist.
#It splits the dataset into training and testing sets and assigns the image data to x_train and x_test variables, and the corresponding labels to y_train and y_test variables.
x_train, x_test = x_train / 255.0, x_test / 255.0
# normalizes the pixel values of the images by dividing them by 255.0. 
# This step scales the pixel values between 0 and 1, making it easier for the neural network to process the data.

plt.figure() # creates a new figure
plt.imshow(x_test[1]) # displays the image located at index 1 
plt.colorbar() # adds a colorbar to the plot, which represents the intensity scale of the image.
plt.grid(False) # turns off the grid
plt.show() # displays the plot 

model = tf.keras.models.Sequential([
    # creates a sequential model using Sequential() function from tf.keras.models
    tf.keras.layers.Flatten(input_shape=(28, 28)), # adds a flatten layer to the model. It transforms the input data, which is a 2D array of shape (28, 28), into a 1D array of shape (784,) to prepare it for the following dense layers.
    tf.keras.layers.Dense(128, activation='relu'), # adds a fully connected dense layer to the model with 128 neurons and ReLU activation function
    tf.keras.layers.Dropout(0.2), # adds a dropout layer to the model with a dropout rate of 0.2
    tf.keras.layers.Dense(10) # adds another fully connected dense layer to the model with 10 neurons
])

prediction1 = model(x_train[:1]).numpy() # prediction using the trained model# .numpy() function is used to convert the TensorFlow tensor to a NumPy array
print(prediction1) # prints the value of predictions1
predictions1 = model(x_test[:1]).numpy() # makes a prediction using the trained model on the first image in the x_test dataset
print(predictions1) # prints the value of predictions1

unmodel = tf.nn.softmax(predictions1).numpy() # applies the softmax function from TensorFlow (tf.nn.softmax()) to the predictions1 array to convert the raw model outputs into probability scores. 
print(unmodel[0]) # prints the value of unmodel[0]


# In[2]:


loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) # creates a loss function called SparseCategoricalCrossentropy using the tf.keras.losses module.
loss = loss_fn(y_train[:1], prediction1).numpy() # calculates the loss value by applying the loss_fn to the ground truth labels y_train[:1] (the first label in the training dataset) and the predicted output prediction1

model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy']) # compiles the model for training. It specifies the optimizer to be used (in this case, 'adam'), the loss function (loss_fn), and the metrics to evaluate during training (accuracy)
model.fit(x_train, y_train, epochs=5) # trains the model using the training dataset x_train and the corresponding labels y_train
model.evaluate(x_test, y_test, verbose=2) # computes the loss value and metrics (accuracy) on the test data.#The verbose=2 argument specifies the verbosity mode, where 2 indicates progress updates are shown for each epoch.

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()]) # creates a new model called probability_model by combining the trained model with an additional Softmax layer
print(probability_model(x_test[:5])) # makes predictions using the probability_model on the first 5 images in the test dataset x_test. It prints the resulting probability distribution for each image

