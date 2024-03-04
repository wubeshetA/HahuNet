import os
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, \
    MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential


class HahuNet:
    """ Convolutional Neural Network (3d) Model for indetifying hand-written
    Amharic letters. 
    """

    def __init__(self, input_shape, num_classes):
        """ Initialize Model

        Args:
            input_shape (tuple): shape of input data (height, width, channels)
            num_classes (int): number of classes to predict
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self._build_model()

    def _build_model(self):
        """ Define the model architecture
        """
        self.model = Sequential([
            Conv2D(16, (3, 3), activation='relu',
                   input_shape=self.input_shape, padding='valid'),
            MaxPooling2D((2, 2)),
            Conv2D(32, (5, 5), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (9, 9), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])

        self.model.compile(
            loss=tf.compat.v1.losses.sparse_softmax_cross_entropy,
            metrics=['accuracy'])

    def train(self, X_train, Y_train, epochs, batch_size):
        """ Train the model

        Args:
            X_train (numpy.array): Training data with shape 
            (num_samples, height, width, channels)
            Y_train (numpy.array): One-hot encoded labels with shape 
            (num_samples, num_classes)
            epochs (int): Number of epochs to train the model
            batch_size (int): Number of samples per gradient update
        """
        self.history = self.model.fit(
            X_train, Y_train, epochs=epochs, batch_size=batch_size,
            validation_split=0.1)

    def evaluate(self, X_test, Y_test):
        """ Evaluate the model

        Args:
            X_test (numpy.ndarray): Test data with shape 
            (num_samples, height, width, channels)
            Y_test (numpy.ndarray): One-hot encoded labels with shape 
            (num_samples, num_classes)

        Returns:
            list: [loss, accuracy]
        """
        return self.model.evaluate(X_test, Y_test)

    def summary(self):
        """ Print model summary
        """
        self.model.summary()
