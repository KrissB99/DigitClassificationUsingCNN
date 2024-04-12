import numpy as np
import os
import matplotlib.pyplot as plt
from colorama import Fore

from mnist_dataset import train_images, train_labels, test_images, test_labels

# Constant values
PLOTS_PATH = './app/dataset/plots'

def create_folder_if_not_exists(plots_path:str = PLOTS_PATH) -> None:
    """Create folder in choosen location if it doesn't already exists

    Args:
        plots_path (str, optional): Folder path. Defaults to PLOTS_PATH.
    """
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)
        print(f"{Fore.GREEN}Folder '{plots_path}' created successfully.{Fore.WHITE}")

def save_image(plt:plt, save:bool = False, images_plot_path:str = 'mnist_img_plot') -> None:
    """Function to save generated plots

    Args:
        save (bool, optional): Save or just show. Defaults to False.
        images_plot_path (str, optional): path to choosen folder. Defaults to 'mnist_img_plot'.
    """
    if save:
        # Check path to choosen directory and create if does not already exists
        create_folder_if_not_exists()
        # Save plot in choosen directory
        plt.savefig(images_plot_path)
        # Response
        if os.path.exists(images_plot_path):
            print(f'{Fore.GREEN}Plot saved in {images_plot_path}.{Fore.WHITE}')
        else:
            print(f'{Fore.RED}Something went wrong!{Fore.WHITE}')
    else:
        plt.show()

def show_images_from_dataset(save:bool = False, folder_path:str = PLOTS_PATH) -> None:
    """Showing first 25 images from dataset

    Args:
        save (bool, optional): Save or show. Defaults to False.
        folder_path (str, optional): path to choosen folder. Defaults to PLOTS_PATH.
    """
    plt.figure(figsize=(6,6))
    plt.subplots_adjust(hspace=0.5, wspace=0.5) 
    
    # Show each image as subplots
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(test_images[i])

    save_image(plt, save, f'{folder_path}/Data_Visualization_digits.png')

def labels_distribution(save: bool = False, folder_path:str = PLOTS_PATH):

    # Count the number of occurrences of each label
    test_images_np, train_counts_np = np.unique(test_images, return_counts=True)
    test_labels_np, test_counts_np = np.unique(test_labels, return_counts=True)

    # Find unique labels and merge them
    all_labels = np.union1d(test_images_np, test_labels_np)

    # Combine counts for shared labels and set 0 for missing labels in train or test
    train_counts_merged = [train_counts_np[test_images_np.tolist().index(label)] if label in test_images_np else 0 for label in all_labels]
    test_counts_merged = [test_counts_np[test_labels_np.tolist().index(label)] if label in test_labels_np else 0 for label in all_labels]

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

    save_image(save, f'{folder_path}/Labels_distribution.png')

def show_image_from_mnist_dataset(img_nr:int, save: bool = False, folder_path:str = PLOTS_PATH) -> None:
    """Showing choosen image from MNIST dataset

    Args:
        img_nr (int): Number of element from the MNIST dataste (train_images)
        save (bool, optional): Save or show plot. Defaults to False.
        folder_path (str, optional): choose where to save generated images. Defaults to PLOTS_PATH.
    """
    plt.imshow(train_images[img_nr,:,:], cmap = plt.cm.binary)
    plt.title(f'Etykieta:  {train_labels[img_nr]}')
    save_image(save, f'{folder_path}/image_{img_nr}.png')

# Uncomment to see the plots

# show_images_from_dataset()
# labels_distribution()
# show_image_from_mnist_dataset(2)
# show_image_from_mnist_dataset(522)
# show_image_from_mnist_dataset(705)