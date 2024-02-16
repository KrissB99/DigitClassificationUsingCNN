import numpy as np
import matplotlib.pyplot as plt

from mnist_dataset import train_images, train_labels, test_labels

def show_images_from_dataset():
    plt.figure(figsize=(6,6))
    plt.subplots_adjust(hspace=0.5, wspace=0.5) 
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(train_labels[i])
    plt.savefig('plots/Data_Visualization_digits.png')

def labels_distribution():

  # Count the number of occurrences of each label
  train_labels_np, train_counts_np = np.unique(train_labels, return_counts=True)
  test_labels_np, test_counts_np = np.unique(test_labels, return_counts=True)

  # Find unique labels and merge them
  all_labels = np.union1d(train_labels_np, test_labels_np)

  # Combine counts for shared labels and set 0 for missing labels in train or test
  train_counts_merged = [train_counts_np[train_labels_np.tolist().index(label)] if label in train_labels_np else 0 for label in all_labels]
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

  plt.savefig('plots/Labels_distribution.png')

# show_images_from_dataset() # see plots/Data_Visualization_digits.png
labels_distribution() # see plots/Labels_distribution.png