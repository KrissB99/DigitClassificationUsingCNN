# External libraries
import numpy as np


# Grain setting for reproducibility
np.random.seed(42)

# List of input data size from 10 000 to 80 000 with 10 000 step
SIZES = list(range(10000, 80000, 10000))

# Setting the input data size, number of epochs and number of classes
EPOCHS, NUM_CLASSES = 10, 10

# Generate a random integer between 0 and 100
SEEDS = [np.random.randint(0, 101) for i in range(0, 5)] 