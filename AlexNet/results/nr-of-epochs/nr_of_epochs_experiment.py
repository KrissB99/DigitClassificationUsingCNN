import torch
from torch.utils.data import DataLoader, Subset
import time

from ...func import create_alexnet_model, set_seed, split_dataset, train_alexnet_on_mnist
from ...const_vars import COMBINED_DATA

SIZE, EPOCHS = 30000, 15

# Ustawienie ziarna dla reprodukowalności
set_seed(42)

# Podział danych w stosunku 80/20
indices = torch.randperm(len(COMBINED_DATA))[:SIZE]
combined_data_sliced = Subset(COMBINED_DATA, indices)
train_dataset, validation_dataset = split_dataset(combined_data_sliced)

# Utworzenie Data Loaders dla zestawu treningowego i testowego
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(validation_dataset, batch_size=64, shuffle=True)

# Utworzenie modelu AlexNet
model, criterion, optimizer = create_alexnet_model()

# Trening model
print('-'*41,f' Trening modelu AlexNet na {EPOCHS} epokach ', '-'*41, '\n')

start_time = time.time()
results = train_alexnet_on_mnist(train_loader, val_loader, EPOCHS)
end_time = time.time()

training_time = end_time - start_time
print(f'Trening ukończony w: {training_time} sekund')