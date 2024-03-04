"""This module reads the images in the dataset and convert them to
useable dataset that can be used to train the model."""

import os
import tensorflow as tf
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


class CharacterImage:
    """CharacterImage reads the images in the dataset and convert them 
    to their pixels value.
    it also reads the numberical label for each image from the file name.
    The characters numberical
    label is represtented in such away that the first character in the
    alphabet is represented as 0 and the second as 1 and it goes until the end
    which the last character is represted as 237"""

    def __init__(self, data_dir):
        """Initializes the CharacterImage class

        Args:
            data_dir (str): The directory where the images are stored
        """
        self.data_dir = data_dir
        self.image_filenames = os.listdir(data_dir)

    def __getitem__(self, idx):
        """Gets the image and its label at the given index in the directory

        Args:
            idx (int): The index of the image in the directory

        Returns:
            tuple: The image tensor and its label
        """
        filename = self.image_filenames[idx]
        image = Image.open(os.path.join(self.data_dir, filename)).convert('L')
        image = np.array(image)
        image = tf.convert_to_tensor(image, dtype=tf.uint8)
        label = int(filename.split('.')[0]) - 1
        return image, label

    def __len__(self):
        """Gets the number of images in the directory

        Returns:
            int: The number of images in the directory
        """
        return len(self.image_filenames)


def create_train_test_data(data_dir):
    """ Create training and test datasets from the images read.

    Args:
        data_dir (str): The directory where the images are stored

    Returns:
        tuple: The training and test datasets
    """
    dataset = CharacterImage(data_dir)
    # Get all the datasets
    all_images = dataset.image_filenames

    # Split the dataset into training and test data
    train_images, test_images = train_test_split(
        all_images, test_size=0.1, random_state=42)

    # Create separate datasets for training and test
    train_dataset = [print("Image to train: ", idx) or dataset[idx] for idx in range(
        len(dataset)) if dataset.image_filenames[idx] in train_images]
    test_dataset = [print("Image to test: ", idx) or dataset[idx] for idx in range(
        len(dataset)) if dataset.image_filenames[idx] in test_images]
    return train_dataset, test_dataset


def split_train_test_data(train_dataset, test_dataset):
    """ Split the train and test datasets into features and labels and 
    convert them to numpy arrays.

    Args:
        train_dataset (list): The training dataset
        test_dataset (list): The test dataset

    Returns:
        tuple: The training and test datasets as numpy arrays
    """
    # Convert train and test datasets to numpy arrays
    X_train_orig, Y_train_orig = zip(*train_dataset)
    X_test_orig, Y_test_orig = zip(*test_dataset)

    # Convert to numpy arrays
    X_train_orig = np.array(X_train_orig)\
        .reshape(-1, X_train_orig[0].shape[0], X_train_orig[0].shape[1], 1)
    X_test_orig = np.array(X_test_orig)\
        .reshape(-1, X_test_orig[0].shape[0], X_test_orig[0].shape[1], 1)

    Y_train_orig = np.array(Y_train_orig).reshape(-1, 1)
    Y_test_orig = np.array(Y_test_orig).reshape(-1, 1)

    return X_train_orig, Y_train_orig, X_test_orig, Y_test_orig


def preprocess_data(X_train_orig, Y_train_orig, X_test_orig, Y_test_orig):
    """ Preprocess the data by normalizing the features and one-hot encoding the labels.

    Args:
        X_train_orig (numpy.ndarray): The training dataset
        Y_train_orig (numpy.narray): The training labels
        X_test_orig (numpy.ndarray): The test dataset
        Y_test_orig (numpy.ndarray): The test labels

    Returns:
        tuple: The preprocessed training and test datasets
    """
    # Normalize the images
    X_train = X_train_orig/255.
    X_test = X_test_orig/255.

    encoder = OneHotEncoder()
    Y_train = encoder.fit_transform(Y_train_orig).toarray()
    Y_test = encoder.transform(Y_test_orig).toarray()


    return X_train, Y_train, X_test, Y_test
