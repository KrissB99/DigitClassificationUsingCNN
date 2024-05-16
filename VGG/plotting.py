# External libraries
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_accuracy_and_loss(history) -> None:
    """Create accuracy and loss plots for the model

    Args:
        history (model): Trained model
    """
    def change_to_percent(elements):
      return [i*100 for i in elements]

    _, axs = plt.subplots(2, 1, figsize=(8, 6))

    # Accuracy
    axs[0].plot(change_to_percent(history.history['accuracy']), label=f'Dokładność testowa')
    axs[0].plot(change_to_percent(history.history['val_accuracy']), label='Dokładność walidacyjna')
    axs[0].set_title(f'Dokładność modelu')
    axs[0].set_ylabel('Dokładność')
    axs[0].set_xlabel('Liczba epok')
    axs[0].legend(loc='lower right')

    # Loss
    axs[1].plot(change_to_percent(history.history['loss']), label='Strata treningowa')
    axs[1].plot(change_to_percent(history.history['val_loss']), label='Strata walidacyjna')
    axs[1].set_title(f'Strata modelu')
    axs[1].set_ylabel('Strata')
    axs[1].set_xlabel('Liczba epok')
    axs[1].legend(loc='upper right')

    plt.tight_layout()
    plt.show()
    
def plot_results(model_df: pd.DataFrame) -> None:
    df_t = model_df.T
    plt.figure(figsize=(10, 6))

    # Accuracy
    plt.plot(df_t.index, df_t["Dokładność treningowa"], '-o', label='Dokładność testowa', color='red')
    plt.plot(df_t.index, df_t["Dokładność walidacyjna"], '-o', label='Dokładność walidacyjna', color='orange')

    # Loss
    plt.plot(df_t.index, df_t["Strata treningowa"], '-o', label='Strata testowa', color='blue')
    plt.plot(df_t.index, df_t["Strata walidacyjna"], '-o', label='Strata walidacyjna', color='green')

    plt.title('Dokładność i strata w zależności od ilości danych wejściowych')
    plt.xlabel('Ilość danych wejściowych')
    plt.ylabel('Wartości')
    plt.legend()

    plt.show()

def custom_confusion_matrix(true_classes, predicted_classes) -> None:
    # Create the confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)

    # Plot the confusion matrix using Seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Etykiety przewidziane przez model')
    plt.ylabel('Etykiety rzeczywiste')
    plt.title('Macierz ')
    plt.show()

def ROC_curve(roc_auc, fpr, tpr):

  plt.figure()
  lw = 2
  colors = plt.cm.rainbow(np.linspace(0, 1, 10))

  plt.plot(fpr[2], tpr[2], color=colors[8], lw=lw, label=f'Krzywa ROC dla klasy 2 (AUC = {roc_auc[2]:.4f})')

  plt.plot([0, 1], [0, 1], 'k--', lw=lw)
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('Wartości False Positive')
  plt.ylabel('Wartości True Positive')
  plt.title('Krzywa ROC dla klasy 2')
  plt.legend(loc="lower right")
  plt.show()