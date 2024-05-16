# External libraries
import keras 
import numpy as np


# Getting MNIST dataset from keras
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

# Combining the training and test sets
IMAGES = np.concatenate((train_images, test_images))
LABELS = np.concatenate((train_labels, test_labels))