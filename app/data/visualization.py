import numpy as np
import os
import matplotlib.pyplot as plt
from colorama import Fore

from mnist_dataset import x_train, x_test, y_test

# Constant values
PLOTS_PATH = './app/data/plots'

# Create folder if not exists
if not os.path.exists(PLOTS_PATH):
    os.makedirs(PLOTS_PATH)
    print(f"{Fore.GREEN}Folder '{PLOTS_PATH}' created successfully.{Fore.WHITE}")

def show_images_from_dataset(folder_path:str = PLOTS_PATH):
    # Create a plot
    plt.figure(figsize=(6,6))
    plt.subplots_adjust(hspace=0.5, wspace=0.5) 
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_train[i], cmap=plt.cm.binary)
        plt.xlabel(x_test[i])
        
    # Save plot in choosen directory
    images_plot_path = f'{folder_path}/Data_Visualization_digits.png'
    plt.savefig(images_plot_path)
    
    # Response
    if os.path.exists(images_plot_path):
        print(f'{Fore.GREEN}Plot saved in {images_plot_path}.{Fore.WHITE}')
    else:
        print(f'{Fore.RED}Something went wrong!{Fore.WHITE}')

def labels_distribution(folder_path:str = PLOTS_PATH):

    # Count the number of occurrences of each label
    x_test_np, train_counts_np = np.unique(x_test, return_counts=True)
    y_test_np, test_counts_np = np.unique(y_test, return_counts=True)

    # Find unique labels and merge them
    all_labels = np.union1d(x_test_np, y_test_np)

    # Combine counts for shared labels and set 0 for missing labels in train or test
    train_counts_merged = [train_counts_np[x_test_np.tolist().index(label)] if label in x_test_np else 0 for label in all_labels]
    test_counts_merged = [test_counts_np[y_test_np.tolist().index(label)] if label in y_test_np else 0 for label in all_labels]

    # Create a bar plot for the label distribution
    plt.figure(figsize=(14, 8))
    bar_width = 0.35
    index = np.arange(len(all_labels))

    # Plot bars
    bars_train = plt.bar(index - bar_width/2, train_counts_merged, bar_width, label=f'Dane treningowe ({np.sum(train_counts_np)})', color='burlywood')
    bars_test = plt.bar(index + bar_width/2, test_counts_merged, bar_width, label=f'Dane testowe ({np.sum(test_counts_np)})', color='green')

    # Add the number of elements above each bar
    for bars in [bars_train, bars_test]:
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval),
                    ha='center', va='bottom', color='black')

    # Set the title, labels and legend
    plt.title('Rozkład etykiet w danych treningowych i testowych zestawu MNIST')
    plt.xlabel('Kategorie przydziału')
    plt.ylabel('Ilość obrazów w bazie danych')
    plt.xticks(index, all_labels)
    plt.legend() 

    # Save plot in choosen directory
    images_plot_path = f'{folder_path}/Labels_distribution.png'
    plt.savefig(images_plot_path)

    # Response
    if os.path.exists(images_plot_path):
        print(f'{Fore.GREEN}Plot saved in {images_plot_path}.{Fore.WHITE}')
    else:
        print(f'{Fore.RED}Something went wrong!{Fore.WHITE}')

# show_images_from_dataset()
# labels_distribution()
