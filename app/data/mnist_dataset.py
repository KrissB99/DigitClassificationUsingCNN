import tensorflow as tf

# Load the MNIST dataset as training and testing sets into separate variables
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Check if split ratio is correct
assert x_train.shape == (60000, 28, 28)
assert x_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)

show_dataset_shape = True

if show_dataset_shape:
    print('MNIST Dataset Shape:')
    print('X_train: ' + str(x_train.shape))
    print('Y_train: ' + str(y_train.shape))
    print('X_test:  '  + str(x_test.shape))
    print('Y_test:  '  + str(y_test.shape))