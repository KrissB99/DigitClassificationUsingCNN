import time
import numpy as np
from tensorflow.keras import models, datasets
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.layers.experimental.preprocessing import Resizing
from datetime import datetime

class VGG19:

    def __init__(self):
        # Data
        (self.x_train, self.y_train), (self.x_test, self.y_test) = datasets.mnist.load_data()
        self.data_preprocessing()
        
        # Model
        self.model = models.Sequential()
        self.resize()
        self.build_model()
        
    def data_preprocessing(self):
        # Convert into 3 channels
        self.x_train=np.dstack([self.x_train] * 3)
        self.x_test=np.dstack([self.x_test] * 3)
        
        # Reshape images
        self.x_train = self.x_train.reshape(-1, 28, 28, 3)
        self.x_test= self.x_test.reshape (-1, 28, 28, 3)
        
    def resize(self):
        # Resizing images to make it more suitable for VGG16-like architecture
        self.model.add(Resizing(48, 48, interpolation="bilinear", input_shape=(28, 28, 1)))
        
    def build_model(self):
        
        # 1st layer
        self.model.add(Conv2D(input_shape=(28, 28, 1), filters=64, kernel_size=(3,3), padding="same", activation="relu"))
        self.model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        
        # 2nd layer
        self.model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
        self.model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        
        # 3rd layer
        self.model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        self.model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        self.model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        self.model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        
        # 4th layer
        self.model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        self.model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        self.model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        self.model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        
        # 5th layer
        self.model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        self.model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        self.model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        self.model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        
        # Classification
        self.model.add(Flatten())
        self.model.add(Dense(units=1024,activation="relu"))
        self.model.add(Dense(units=1024,activation="relu"))
        self.model.add(Dense(units=10, activation="softmax"))
        
        self.model.summary()
        
    def train(self, x_train, y_train, n:int = 10000, bach_size:int = 128, epochs:int = 10, verbose: int = 0):
        
        start_time = time.time()  # Start measuring time
        
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # compile model
        history = self.model.fit(x_train[:n], y_train[:n], batch_size=128, epochs=10, verbose=0)
        
        training_time = time.time() - start_time # Stop measuring time
        
        return history, training_time
        
    def save(self):
        self.model.save(f'trained_models/{self.name}_{datetime.now()}')