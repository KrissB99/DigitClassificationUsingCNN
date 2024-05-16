from collections import defaultdict
from torch.utils.data import DataLoader, Subset
import time 
import torch

from ...const_vars import SEEDS, COMBINED_DATA, SIZES
from ...func import set_seed, split_dataset, create_alexnet_model, train_alexnet_on_mnist

output_data = defaultdict()

for (i, size) in enumerate(SIZES):

    # Ustawienie ziarna dla reprodukowalności
    set_seed(SEEDS[i])

    # Podział danych w stosunku 80/20
    indices = torch.randperm(len(COMBINED_DATA))[:size]
    combined_data_sliced = Subset(COMBINED_DATA, indices)
    train_dataset, validation_dataset = split_dataset(combined_data_sliced)

    # Utworzenie Data Loaders dla zestawu treningowego i testowego
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(validation_dataset, batch_size=64, shuffle=True)

    # Utworzenie modelu AlexNet
    model, criterion, optimizer = create_alexnet_model()

    # Trening model
    print('-'*41, f' AlexNet | {size} ', '-'*41, '\n')

    start_time = time.time()
    output_dict = train_alexnet_on_mnist(train_loader, val_loader)
    end_time = time.time()

    training_time = end_time - start_time
    print(f'Trening ukończony w: {training_time} sekund')

    output_data[size] = [training_time]
    for i in output_dict: output_data[size].append(i)

    del indices, combined_data_sliced, train_loader, val_loader, model