import matplotlib.pyplot as plt
import seaborn as sns
from pandas import pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

from func import DEVICE, test_class_probabilities

def plot_results(model_df: pd.DataFrame) -> None:
    # Transpozycja DataFrame dla łatwiejszego plotowania
    df_t = model_df.T

    # Tworzenie wykresu
    plt.figure(figsize=(10, 6))

    # Plotowanie dokładności
    plt.plot(df_t.index, df_t["Dokładność treningowa"], '-o', label='Dokładność treningowa', color='red')
    plt.plot(df_t.index, df_t["Dokładność walidacyjna"], '-o', label='Dokładność walidacyjna', color='orange')

    # Plotowanie straty
    plt.plot(df_t.index, df_t["Strata treningowa"], '-o', label='Strata treningowa', color='blue')
    plt.plot(df_t.index, df_t["Strata walidacyjna"], '-o', label='Strata walidacyjna', color='green')

    # Dodanie tytułu i etykiet
    plt.title('Dokładność i strata w zależności od ilości danych wejściowych')
    plt.xlabel('Ilość danych wejściowych')
    plt.ylabel('Wartości [%]')

    # Dodanie legendy
    plt.legend()

    # Pokazanie wykresu
    plt.show()
  
def plot_accuracy_and_loss(data) -> None:
    """ Tworzenie wykresów dokładności i straty dla modelu

    Args:
        history (model): Wytrenowany model
    """
    # Wykresy dokładności i straty
    _, axs = plt.subplots(2, 1, figsize=(8, 6))

    # Wykres dokładności modelu
    axs[0].plot(data['accuracy'], label=f'Dokładność treningowa')
    axs[0].plot(data['val_accuracy'], label='Dokładność walidacyjna')
    axs[0].set_title(f'Dokładność modelu')
    axs[0].set_ylabel('Dokładność [%]')
    axs[0].set_xlabel('Liczba epok')
    axs[0].legend(loc='lower right')
    axs[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.3f}'))

    # Wykres starty modelu
    axs[1].plot(data['loss'], label='Strata treningowa')
    axs[1].plot(data['val_loss'], label='Strata walidacyjna')
    axs[1].set_title(f'Strata modelu')
    axs[1].set_ylabel('Strata [%]')
    axs[1].set_xlabel('Liczba epok')
    axs[1].legend(loc='upper right')
    axs[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.3f}'))

    plt.tight_layout()
    plt.show()
    
def plot_confusion_matrix(true_labels, predictions, class_names):
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Etykiety przewidziane przez model')
    plt.ylabel('Etykiety rzeczywiste')
    plt.title('Macierz Pomyłek')
    plt.show()

def plot_ROC(model, test_loader, which_class=2, device=DEVICE):
    actuals, class_probabilities = test_class_probabilities(model, device, test_loader, which_class)

    fpr, tpr, _ = roc_curve(actuals, class_probabilities)
    roc_auc = auc(fpr, tpr)
    lw = 2
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='Krzywa ROC (obszar = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Wartości False Positive')
    plt.ylabel('Wartości True Positive')
    plt.title(f'Krzywa ROC dla klasy {which_class}')
    plt.legend(loc="lower right")
    plt.show()