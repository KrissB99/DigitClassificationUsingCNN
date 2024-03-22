from keras.datasets import mnist

# Load the MNIST dataset as training and testing sets into separate variables
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Check if split ratio is correct
assert train_images.shape == (60000, 28, 28)
assert test_images.shape == (10000, 28, 28)
assert train_labels.shape == (60000,)
assert test_labels.shape == (10000,)

# Show shapes of each parameter from mnist dataset
show_dataset_shape = False
if show_dataset_shape:
    print('MNIST Dataset Shape:')
    print('train_images: ' + str(train_images.shape))
    print('train_labels: ' + str(train_labels.shape))
    print('test_images:  '  + str(test_images.shape))
    print('test_labels:  '  + str(test_labels.shape))