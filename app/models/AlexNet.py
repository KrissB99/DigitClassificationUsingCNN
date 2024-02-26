from datetime import datetime
import time
from tensorflow import keras, pad, expand_dims, repeat
from tensorflow.keras import losses, models, datasets
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers.experimental.preprocessing import Resizing


class AlexNetModel:
    
    def __init__(self):
        # Data
        (self.x_train, self.y_train), (self.x_test, self.y_test) = datasets.mnist.load_data()
        self.data_preprocessing()
        
        # Model
        self.model = models.Sequential()
        self.resize()
        self.build_model()
        
    def data_preprocessing(self):
        self.x_train = pad(self.x_train, [[0, 0], [2,2], [2,2]])/255
        x_test = pad(x_test, [[0, 0], [2,2], [2,2]])/255

        self.x_train = expand_dims(self.x_train, axis=3, name=None)
        self.x_test = expand_dims(self.x_test, axis=3, name=None)

        self.x_train = repeat(self.x_train, 3, axis=3)
        self.x_test = repeat(self.x_test, 3, axis=3)

        self.x_val = self.x_train[-2000:,:,:,:]
        self.y_val = self.y_train[-2000:]
        self.x_train = self.x_train[:-2000,:,:,:]
        self.y_train = self.y_train[:-2000]
        
    def resize(self):
        # Resizing images to make it more suitable for AlexNet-like architecture
        self.model.add(Resizing(227, 227, interpolation="bilinear", input_shape=(28, 28, 1)))
        
    def build_model(self):
        
        # First Convolutional Layer
        self.model.add(Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D((3, 3), strides=(2, 2)))
        
        # Second Convolutional Layer
        self.model.add(Conv2D(filters=256, kernel_size=(5, 5), activation='relu', padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D((3, 3), strides=(2, 2)))
        
        # Remaining Convolutional Layers
        self.model.add(Conv2D(filters=384, kernel_size=(3, 3), activation='relu', padding='same'))
        self.model.add(Conv2D(filters=384, kernel_size=(3, 3), activation='relu', padding='same'))
        self.model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same'))
        self.model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        
        self.model.add(Flatten())
        
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dropout(0.5))
        
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dropout(0.5))
        
        self.model.add(Dense(10, activation='softmax'))
        
        self.model.summary()
        
    def train(self, n:int = 10000, bach_size:int = 64, epochs:int = 10, verbose: int = 0):
        start_time = time.time()
        
        self.model.compile(optimizer='adam', loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])
        self.history = self.model.fit(self.x_train[:n], self.y_train[:n], batch_size=bach_size, epochs=epochs, verbose=0)
        
        training_time = time.time() - start_time # Stop measuring time
        self.save()
        
        return self.history, training_time
    
    def save(self):
        self.model.save(f'trained_models/AlexNet_{datetime.now()}')
        
    def evaluate(self):
        self.model.evaluate(self.x_test, self.y_test)
        
model = AlexNetModel()
model.nn_compile()