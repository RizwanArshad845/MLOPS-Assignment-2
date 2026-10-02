import os

import numpy as np
import yaml
from sklearn.model_selection import train_test_split

with open("params.yaml") as f:
    params = yaml.safe_load(f)["preprocess"]

raw = np.load("data/raw/fashion_mnist_raw.npz")
x_train, y_train = raw["x_train"], raw["y_train"]
x_test, y_test = raw["x_test"], raw["y_test"]

# normalize pixel values to [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_tr, x_val, y_tr, y_val = train_test_split(
    x_train, y_train,
    test_size=params["val_size"],
    random_state=params["seed"],
    stratify=y_train,
)

os.makedirs("data/processed", exist_ok=True)
np.savez_compressed("data/processed/train.npz", x=x_tr, y=y_tr)
np.savez_compressed("data/processed/val.npz", x=x_val, y=y_val)
np.savez_compressed("data/processed/test.npz", x=x_test, y=y_test)
print("Saved processed data:", x_tr.shape, x_val.shape, x_test.shape)