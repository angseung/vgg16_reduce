import os
from typing import Any
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from image_dataset import image_dataset_from_directory


def gen_datasets(path: str = "D:/imagenet-mini",
                 batch_size: int= 64,
                 shuffle: bool = True,
                 seed: Any = None,
                 ) -> tf.data.Dataset:

    # dataset = tf.keras.utils.image_dataset_from_directory(
    dataset = image_dataset_from_directory(
        path,
        labels='inferred',
        label_mode="categorical",
        # class_names=None,
        color_mode='rgb',
        batch_size=batch_size,
        image_size=(224, 224),
        shuffle=shuffle,
        seed=seed,
        # validation_split=None,
        # subset=None,
        # interpolation='bilinear',
        # follow_links=False,
        # crop_to_aspect_ratio=False,
    )

    return dataset
