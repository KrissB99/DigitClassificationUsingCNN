# External libraries
from collections import defaultdict
import keras
import numpy as np

# Internal libraries
from data import IMAGES, LABELS
from const_vars import EPOCHS, SEEDS, SIZES
from func import area_under_the_curve, build_VGG_model, compute_metrics, create_df, predict, resize_images, shuffle_data, train_model, input_data_analysis
from plotting import ROC_curve, custom_confusion_matrix, plot_accuracy_and_loss, plot_results

def main_experiment(i: int, vgg_model_type: str, epochs: int, size: int):

    output_data = defaultdict()

    sampled_images, sampled_labels = shuffle_data(IMAGES, LABELS, SEEDS[i], size)

    # Data pre-processing
    images_resized = resize_images(sampled_images, [32, 32])
    y_one_hot = keras.utils.to_categorical(sampled_labels, 10)

    # Split data into 80/20 ratio
    split_index = int(len(images_resized) * 0.8)
    x_train, x_test = images_resized[:split_index], images_resized[split_index:]
    y_train, y_test = y_one_hot[:split_index], y_one_hot[split_index:]

    VGG_model = build_VGG_model(vgg_model_type)

    # Model training
    history, training_time = train_model(model = VGG_model,
                                            x_train = x_train,
                                            y_train = y_train,
                                            val_data = (x_test, y_test),
                                            epochs = epochs)

    # Accuracy and loss
    plot_accuracy_and_loss(history)

    # Model testing
    predictions, predicting_time  = predict(VGG_model, x_test)

    # Convert predictions and y_test to class labels
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)

    output_data[i] = [training_time,
                        predicting_time,
                        history.history['accuracy'][-1],
                        history.history['val_accuracy'][-1],
                        history.history['loss'][-1],
                        history.history['val_loss'][-1]]

    # Confusion matrix
    custom_confusion_matrix(true_classes, predicted_classes)

    # AUC and ROC
    roc_auc, fpr, tpr = area_under_the_curve(predictions)
    ROC_curve(roc_auc, fpr, tpr)

    # Evaluation metrics
    compute_metrics(true_classes, predicted_classes)
    
def data_volume_experiment(vgg_model_type: str):
    output_data_vgg = defaultdict()
    for (i, size) in enumerate(SIZES):
        
        print('-'*41, f'Ilość danych wejściowych: {size}, model: {vgg_model_type}', '-'*41, '\n')
        
        history_vgg, training_time_vgg = input_data_analysis(vgg_model_type, size, IMAGES, LABELS, SEEDS[i], EPOCHS)

        output_data_vgg[size] = [training_time_vgg,
                                history_vgg.history['accuracy'][-1]*100,
                                history_vgg.history['val_accuracy'][-1]*100,
                                history_vgg.history['loss'][-1]*100,
                                history_vgg.history['val_loss'][-1]*100]

        del history_vgg, training_time_vgg
        
    vgg_df = create_df(output_data_vgg)

    # Table
    print(vgg_df)
    # Results
    plot_results(vgg_df)
    
def nr_of_epochs_experiment(vgg_model_type: str, choosen_size: int, epochs: int):
    history_vgg, _ = input_data_analysis(vgg_model_type, choosen_size, IMAGES, LABELS, SEEDS[0], epochs)
    plot_accuracy_and_loss(history_vgg)