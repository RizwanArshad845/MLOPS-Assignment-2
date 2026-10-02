import os

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from tensorflow import keras

with open("params.yaml") as f:
    params = yaml.safe_load(f)["train"]

tf.keras.utils.set_random_seed(params["seed"])

train = np.load("data/processed/train.npz")
val = np.load("data/processed/val.npz")

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(params["dense_units"], activation="relu"),
    keras.layers.Dropout(params["dropout_rate"]),
    keras.layers.Dense(10, activation="softmax"),
])
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=params["learning_rate"]),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(
    train["x"], train["y"],
    validation_data=(val["x"], val["y"]),
    epochs=params["epochs"],
    batch_size=params["batch_size"],
)

os.makedirs("models", exist_ok=True)
model.save("models/model.h5")
pd.DataFrame(history.history).to_csv("models/history.csv", index=False)
print("Saved models/model.h5 and models/history.csv")