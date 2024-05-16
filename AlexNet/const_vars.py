# External libraries
import numpy as np
import torch
from torch.utils.data import ConcatDataset

# Internal libraries
from data import mnist_train, mnist_test


# Merging the test and training sets
COMBINED_DATA = ConcatDataset([mnist_train, mnist_test])

# Grain setting for reproducibility
np.random.seed(42)

# Getting device type (CUDA or CPU) 
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Setting the input data size, number of epochs and number of classes
EPOCHS, NUM_CLASSES = 15, 10

# List of input data size from 10 000 to 80 000 with 10 000 step
SIZES = list(range(10000, 80000, 10000))

# Generowanie losowej liczby całkowitej z przedziału od 0 do 100
SEEDS = [np.random.randint(0, 101) for i in range(0, 5)] 