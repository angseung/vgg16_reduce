import os
from typing import Any

import torch.nn

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from image_dataset import image_dataset_from_directory


def gen_datasets(
    path: str = "D:/imagenet-mini",
    batch_size: int = 64,
    shuffle: bool = True,
    seed: Any = None,
) -> tf.data.Dataset:

    # dataset = tf.keras.utils.image_dataset_from_directory(
    dataset = image_dataset_from_directory(
        path,
        labels="inferred",
        label_mode="categorical",
        # class_names=None,
        color_mode="rgb",
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


def get_flatten_model(model: torch.nn.Module = None) -> list:
    layers = []

    for i, block in enumerate(model.children()):
        if block.__class__.__name__ == "Sequential":
            for layer in block.children():
                layers.append(layer)
        else:
            layers.append(block)

    o_layer = layers[0:32] + [torch.nn.Flatten(start_dim=1)] + layers[32:]

    return o_layer


def get_flatten_model_resnet50(model: torch.nn.Module = None) -> list:
    layers = []

    for i, block in enumerate(model.children()):
        layers.append(block)

    return o_layer


def get_trimed_model(model, start_cut: int = 0, end_cut: int = None):
    layers = get_flatten_model(model)

    if end_cut:
        assert end_cut < len(layers)
        return torch.nn.Sequential(*layers[start_cut : end_cut + 1])
    else:
        return torch.nn.Sequential(*layers[start_cut:])


def channel_repeat(images: np.ndarray = None, out_channel: int = 64) -> np.ndarray:
    images = np.array(images)
    in_channel = images.shape[1]
    n_repeat, n_else = out_channel // in_channel, out_channel % in_channel
    repeat_array = np.tile([0, 1, 2], reps=n_repeat + 1)
    re_images = np.zeros(
        [images.shape[0], out_channel, images.shape[2], images.shape[3]]
    )

    for i, val in enumerate(repeat_array):
        re_images[:, i, :, :] = images[:, val, :, :]

        if i == out_channel - n_else:
            break

    return re_images


# a = np.array((range(3*5*5))).reshape(1, 3, 5, 5)
# b = channel_repeat(a, 100)
