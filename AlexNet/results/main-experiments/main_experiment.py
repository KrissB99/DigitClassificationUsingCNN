import torch
from torch.utils.data import DataLoader, Subset
import time
from collections import defaultdict

# Internal libraries
from ...const_vars import COMBINED_DATA, SIZE, EPOCHS, SEEDS, NUM_CLASSES
from ...plotting import plot_ROC, plot_accuracy_and_loss, plot_confusion_matrix
from ...func import compute_auc_per_class, \
                    compute_precision_recall_f1score, \
                    create_alexnet_model, set_seed, \
                    split_dataset, train_alexnet_on_mnist, \
                    validate_model

SIZE = 30000 

for i in range(1, 5):

    # Grain setting for reproducibility
    set_seed(SEEDS[i])

    # Split dataset
    indices = torch.randperm(len(COMBINED_DATA))[:SIZE]
    combined_data_sliced = Subset(COMBINED_DATA, indices)
    train_dataset, validation_dataset = split_dataset(combined_data_sliced)

    # Created Data Loaders for the training and test set
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(validation_dataset, batch_size=64, shuffle=True)

    # Creation of the AlexNet model
    model, criterion, optimizer = create_alexnet_model()

    # Train model
    start_time = time.time()
    output_dict = train_alexnet_on_mnist(model=model, criterion=criterion, optimizer=optimizer, train_loader=train_loader, val_loader=train_loader, epochs=EPOCHS)
    end_time = time.time()

    training_time = end_time - start_time
    print(f'Trening ukończony w: {training_time} sekund')

    # Accuracy, Loss for training and validation
    metrics = defaultdict(list)
    for val in output_dict.values():
        metrics['accuracy'].append(val[0])
        metrics['val_accuracy'].append(val[1])
        metrics['loss'].append(val[2])
        metrics['val_loss'].append(val[3])

    plot_accuracy_and_loss(metrics)

    # Model testing
    v_start_time = time.time()
    predictions, true_labels = validate_model(model, val_loader)
    v_end_time = time.time()

    val_time = v_end_time - v_start_time
    print(f'Trening ukończony w: {val_time} sekund')

    # Confusion matrix
    class_names = [str(i) for i in range(10)]
    plot_confusion_matrix(true_labels, predictions, class_names)

    # AUC and ROC
    plot_ROC(model, val_loader)

    aucs = compute_auc_per_class(model, val_loader)

    for x, a in aucs.items():
        print(f'AUC dla klasy {x}: {round(a, 3)}')

    # Evaluation metrics
    precision, recall, f1score = compute_precision_recall_f1score(true_labels, predictions, NUM_CLASSES)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1score)