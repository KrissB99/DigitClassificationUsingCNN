from torch.utils.data import random_split
from torchvision.models import alexnet
import torch.optim as optim
import torch.nn as nn
import torch
from collections import defaultdict
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support
import random
import pandas as pd
import numpy as np

# Internal libraries
from const_vars import DEVICE

# DATASET PREPROCESSING

def create_df(output_data: dict) -> pd.DataFrame:
    INDEXES = ["Czas treningu", "Dokładność treningowa", "Dokładność walidacyjna", "Strata treningowa", "Strata walidacyjna"]
    return pd.DataFrame(output_data, index=INDEXES)

def split_dataset(dataset) -> tuple:
    """Split dataset into training and test dataset in 80/20 ratio

    Args:
        dataset: _description_

    Returns:
        tuple: (train_dataset, validation_dataset)
    """
    # Obliczanuie długości dla zbioru testowego i treningowego w stosunku 80/20
    train_size = int(0.8 * len(dataset))
    validation_size = len(dataset) - train_size

    # Podzielenie na dwa zestawy danych
    train_ds, validation_ds = random_split(dataset, [train_size, validation_size])
    return train_ds, validation_ds

# ALEXNET MODEL FUNCTIONS

def set_seed(seed_value: int = 42):
    """Set the seed of randomness for repeatability of results

    Args:
        seed_value (int, optional): Ziarno. Defaults to 42.
    """
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed_value)
    random.seed(seed_value)
    
def create_alexnet_model() -> tuple:
    """Implementation of alexnet pretrained model from pytorch"""
    model = alexnet(weights=True)
    # Dostosowanie do 10 klas dla zestawu MNIST
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, 10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    return model, criterion, optimizer

def train_alexnet_on_mnist(model, criterion, optimizer, train_loader, val_loader, epochs=10, device=DEVICE) -> list:
    output_data_alexnet = defaultdict(list)
    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total

        # Validation
        model.eval()
        running_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_loss = running_loss / len(val_loader)
        test_accuracy = 100 * correct / total

        output_data_alexnet[epoch] = [train_accuracy, test_accuracy, train_loss, test_loss]
        print(f'Epoch {epoch+1}/{epochs} | loss: {train_loss:.2f}% - accuracy: {train_accuracy:.2f}%, val_loss: {test_loss:.2f}% - val_accuracy: {test_accuracy:.2f}%')

    return output_data_alexnet

def validate_model(model, test_loader, device = DEVICE):
    """Validate AlexNet model on test data"""
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    return np.array(predictions), np.array(true_labels)

def test_class_probabilities(model, device, test_loader, which_class):
    model.eval()
    actuals = []
    probabilities = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            prediction = output.argmax(dim=1, keepdim=True)
            actuals.extend((target.view_as(prediction) == which_class).cpu().numpy())
            probabilities.extend(torch.exp(output[:, which_class]).cpu().numpy())
    return actuals, probabilities

# MODEL EVALUATION

def sanitize_inputs(inputs):
    inputs = np.nan_to_num(inputs, nan=0.0, posinf=1.0, neginf=0.0)
    return inputs

def compute_auc_per_class(model, test_loader, device=DEVICE):
  aucs = defaultdict()

  for i_class in range(0, 10):
    actuals, class_probabilities = test_class_probabilities(model, device, test_loader, i_class)
    actuals = sanitize_inputs(actuals)
    class_probabilities = sanitize_inputs(class_probabilities)
    fpr, tpr, _ = roc_curve(actuals, class_probabilities)
    aucs[i_class] = auc(fpr, tpr)

  return aucs

def compute_precision_recall_f1score(true_labels, predictions, num_classes):
    precision, recall, f1score, _ = precision_recall_fscore_support(true_labels, predictions, average='macro')
    return precision, recall, f1score