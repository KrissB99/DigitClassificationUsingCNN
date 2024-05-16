# External libraries
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import keras
from keras.applications import VGG16, VGG19
from keras import layers, models
import tensorflow as tf
import pandas as pd
import numpy as np
import time


# DATA PREPROCESSING

def shuffle_data(images: np.ndarray, labels: np.ndarray, random_seed: int, size: int) -> list:
    """Shuffle data for cross-validation

    Args:
        images (np.ndarray): Input images
        labels (np.ndarray): labels from dataset
        random_seed (int): The seed is set so that the results can be reproduced
        size (int): 

    Returns:
        (tuple): (sampled_images, sampled_labels)
    """
    # Zapewnienie odtwarzalności
    np.random.seed(random_seed)

    # Tasowanie indeksów w celu losowego wyboru danych
    indices = np.arange(images.shape[0])
    np.random.shuffle(indices)

    # Wybór określonej liczby próbek
    selected_indices = indices[:size]

    return images[selected_indices], labels[selected_indices]

def resize_images(images: np.ndarray, shape:list) -> tf.Tensor:
  """Converting images to 3-channel RGB type,
       resizing input images to fit the selected model and
       normalization of images to values ​​​​in the range [0, 255]

  Args:
      images (np.ndarray): Input images
      shape (list): Required shape

  Returns:
      images_resized: tf.Tensor
  """
  # Transform to RGB (3 channels)
  images_rgb = np.repeat(images[..., np.newaxis], 3, axis=-1)
  # Resizing
  images_resized = tf.image.resize(images_rgb, shape)
  # Normalization
  images_resized /= 255.0

  return images_resized

def create_df(output_data: dict) -> pd.DataFrame:
    INDEXES = ["Czas treningu", "Czas klasyfikacji", "Dokładność treningowa", "Dokładność walidacyjna", "Strata treningowa", "Strata walidacyjna"]
    return pd.DataFrame(output_data, index=INDEXES)

# MODEL IMPLEMENTATION, TESTING AND VALIDATION

def build_VGG_model(model_type: str):
  # Create a VGG16 model pre-trained on ImageNet data, excluding the top fully connected layers.
  if model_type == 'VGG16':
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
  elif model_type == 'VGG19':
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
  else:
    raise ValueError("Wrong value! model value must be 'VGG16' or 'VGG19'")

  # Freezing the base model layers to prevent them from updating during the first training phase
  for layer in base_model.layers:
      layer.trainable = False

  model = models.Sequential()
  model.add(base_model)

  model.add(layers.Flatten())
  model.add(layers.Dense(256, activation='relu'))
  model.add(layers.Dropout(0.5))
  model.add(layers.Dense(10, activation='softmax')) # 10 klas dla klasyfikacji cyfr

  model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

  model.summary()

  return model

def train_model(model, x_train, y_train, val_data, epochs) -> list:
  """Model training on input data, with time measurement

  Args:
      model (keras.models): Model architecture
      x_train (list): Train dataset 

  Returns:
      images_resized: tf.Tensor
  """
  # Start of time measurement
  start_time = time.time()

  # Model training
  history = model.fit(x_train, y_train,
                      epochs=epochs,
                      batch_size=64,
                      validation_data=val_data,
                      verbose=1)

  # Stop of time measurement
  end_time = time.time()
  training_time = end_time - start_time
  print(f"Trening ukończony w przeciągu: {training_time:.2f} sekund")

  return history, training_time

def predict(model, x_test):
  start_time = time.time()

  # Make predictions on the test set
  predictions = model.predict(x_test)

  end_time = time.time()
  predicting_time = end_time - start_time
  print(f"Prediction completed in: {predicting_time:.2f} seconds")

  return predictions, predicting_time

# EVALUATION METRICS

def area_under_the_curve(y_test, predictions):

  # Binarize the labels in a one-vs-all fashion
  n_classes = y_test.shape[1]
  y_test_binarized = label_binarize(y_test, classes=[*range(n_classes)])

  # Compute ROC curve and ROC area for each class
  fpr = dict()
  tpr = dict()
  roc_auc = dict()
  for i in range(n_classes):
      fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], predictions[:, i])
      roc_auc[i] = auc(fpr[i], tpr[i])

  # Optionally, compute the micro-average ROC curve and ROC area
  fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarized.ravel(), predictions.ravel())
  roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

  for i in range(n_classes):
    print(f'AUC for class {i}:', round(roc_auc[i]*100, 3))

  return roc_auc, fpr, tpr

def compute_metrics(true_classes, predicted_classes):

  # Precision
  precision = precision_score(true_classes, predicted_classes, average='macro')
  print(f'Precision: {precision}')

  # Recall
  recall = recall_score(true_classes, predicted_classes, average='macro')
  print(f'Recall: {recall}')

  # F1 Score
  f1 = f1_score(true_classes, predicted_classes,  average='macro')
  print(f'F1 Score: {f1}')
  
# INPUT DATA ANALYSIS

def input_data_analysis(model: str, size: int, images, labels, seed: int, epochs: int) -> list:
  """Input data adjustment, model implementation and training

  Args:
      model (str): VGG16 or VGG19
      size (int): Number of input data
      seed (int): Ziarno reprodukowalności,
      epochs (int): Training epochs number

  Returns:
      (list): trained model, training time
  """

  # Wybór odpowiedniej ilosci danych
  sampled_images, sampled_labels = shuffle_data(images, labels, seed, size)

  # Data pre-processing
  images_resized = resize_images(sampled_images, [32, 32])

  # Label processing
  y_one_hot = keras.utils.to_categorical(sampled_labels, 10)

  # 80/20 ratio splitting
  split_index = int(len(images_resized) * 0.8)
  x_train, x_test = images_resized[:split_index], images_resized[split_index:]
  y_train, y_test = y_one_hot[:split_index], y_one_hot[split_index:]

  VGG_model = build_VGG_model(model)

  # Model training
  history, training_time = train_model(model = VGG_model,
                                      x_train = x_train,
                                      y_train = y_train,
                                      val_data = (x_test, y_test),
                                      epochs = epochs)

  return history, training_time