import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

path = "D:/imagenet-mini"
labels = os.listdir(path + "/train")
test_labels = os.listdir(path + "/val")

# train_data = tfds.ImageFolder(path, shape=(224, 224, 3)).as_dataset(split='train', shuffle_files=True)
# test_data = tfds.ImageFolder(path, shape=(224, 224, 3)).as_dataset(split='val', shuffle_files=True)
train_data = tf.keras.utils.image_dataset_from_directory(
    path, labels='inferred', label_mode='int',
    class_names=None, color_mode='rgb', batch_size=32, image_size=(256,
    256), shuffle=True, seed=None, validation_split=None, subset=None,
    interpolation='bilinear', follow_links=False,
    crop_to_aspect_ratio=False, **kwargs
)


assert labels == test_labels

model = tf.keras.applications.vgg16.VGG16(
    include_top=True,
    weights="imagenet",
    input_shape=(224, 224, 3),
    pooling=None,
    classes=1000,
)
model.trainable = False

for layer in model.layers:
    # print(layer.input)
    print(layer)

reduced_model = keras.Sequential(
    [keras.layers.Conv2D(32, kernel_size=3, strides=1, padding="same")] + model.layers
)

model.evaluate()