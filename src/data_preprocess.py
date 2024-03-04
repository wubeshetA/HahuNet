"""This module reads the images in the dataset and convert them to
useable dataset that can be used to train the model."""

import os
import tensorflow as tf
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


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


def load_train_test_data(data_dir):
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
    train_dataset = [dataset[idx] for idx in range(
        len(dataset)) if dataset.image_filenames[idx] in train_images]
    test_dataset = [dataset[idx] for idx in range(
        len(dataset)) if dataset.image_filenames[idx] in test_images]
    return train_dataset, test_dataset
