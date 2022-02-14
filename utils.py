import os
from typing import Any
import torch.nn
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from PIL import Image
from torchvision.datasets import ImageFolder
from image_dataset import image_dataset_from_directory
from carve import resize


class Resize:
    def __init__(self, size, aspect='wide'):
        if isinstance(size, int):
            self.target_size = size
        elif isinstance(size, tuple):
            raise ValueError('Size must be integer type')

        self.aspect = aspect  # 'wide' or 'narrow'

    def __call__(self, img):
        # if not isinstance(img, np.ndarray):
        #     image = np.array(img)
        # input must be PIL image type

        w, h = img.size
        if w < h:
            if self.aspect == 'narrow':
                ow = int(self.target_size * w / h)
                oh = self.target_size
            else:
                ow = self.target_size
                oh = int(self.target_size * h / w)
        else:
            if self.aspect == 'narrow':
                ow = self.target_size
                oh = int(self.target_size * h / w)

            else:
                ow = int(self.target_size * w / h)
                oh = self.target_size

        return img.resize((ow, oh))


class SeamCarvingResize:
    def __init__(self, size, energy_mode="backward"):
        if isinstance(size, int):
            self.target_size = (size, size)
        elif isinstance(size, tuple):
            if len(size) == 2:
                self.target_size = size
            else:
                raise ValueError("Size must be int or tuple with length 2 (h, w)")

        self.energy_mode = energy_mode  # 'forward' or 'backward'

    def __call__(self, img):
        if not isinstance(img, np.ndarray):
            image = np.array(img)

        # dst = seam_carving.resize(
        dst = resize(
            image,
            (self.target_size[0], self.target_size[1]),
            energy_mode=self.energy_mode,  # Choose from {backward, forward}
            order="width-first",  # Choose from {width-first, height-first}
            keep_mask=None,
        )

        return Image.fromarray(dst)

    def __repr__(self):
        return self.__class__.__name__ + "(target_size=%d, energe_mode=%s)" % (
            self.target_size[0],
            self.energy_mode,
        )


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

# dataset = ImageFolder(root="E:/imagenet-mini/train")