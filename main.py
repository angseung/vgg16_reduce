import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import pickle
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import preprocess_input
from utils import gen_datasets


# path = "C:/imagenet-mini"
path = "C:/imagenet"

train_data = gen_datasets(
    path + "/train",
    batch_size=32,
    shuffle=True,
    seed=None,
)
test_data = gen_datasets(
    path + "/val",
    batch_size=64,
    shuffle=True,
    seed=None,
)

model = tf.keras.applications.vgg16.VGG16(
    include_top=True,
    weights="imagenet",
    input_shape=(224, 224, 3),
    pooling=None,
    classes=1000,
)
# model.trainable = False

# base_learning_rate = 0.0001

# original VGG model...
# model.compile(
#     optimizer=tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate),
#     loss="categorical_crossentropy",
#     metrics=["accuracy"],
# )
model.evaluate(test_data, batch_size=32)
#
# ## reduced model...
# reduced_model = keras.Sequential(
#     [keras.layers.Conv2D(64, kernel_size=3, strides=2, padding="same")]
#     + model.layers[4:]
# )
#
# reduced_model.compile(
#     optimizer=tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate),
#     loss="categorical_crossentropy",
#     metrics=["accuracy"],
# )
# reduced_model.fit(train_data, batch_size=32, epochs=10)
# reduced_model.evaluate(test_data, batch_size=32)
