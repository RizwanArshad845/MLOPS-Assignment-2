import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from tensorflow import keras

test = np.load("data/processed/test.npz")
model = keras.models.load_model("models/model.h5")

loss, acc = model.evaluate(test["x"], test["y"], verbose=0)
pred = np.argmax(model.predict(test["x"], verbose=0), axis=1)

os.makedirs("reports", exist_ok=True)
ConfusionMatrixDisplay(confusion_matrix(test["y"], pred)).plot(cmap="Blues")
plt.savefig("reports/confusion_matrix.png", dpi=120, bbox_inches="tight")

with open("metrics.json", "w") as f:
    json.dump({"test_loss": float(loss), "test_accuracy": float(acc)}, f, indent=2)
print(f"test_loss={loss:.4f} test_accuracy={acc:.4f}")