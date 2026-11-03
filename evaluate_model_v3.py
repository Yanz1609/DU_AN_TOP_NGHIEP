
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

MODEL_PATH = "models/eye_cnn_v3.keras"
TEST_DIR = "data/test"
IMAGE_SIZE = (96, 96)

model = tf.keras.models.load_model(MODEL_PATH)

images = []
labels = []

classes = {
    "closed_eyes": 0,
    "open_eyes": 1,
}

for folder, label in classes.items():
    folder_path = os.path.join(TEST_DIR, folder)

    for filename in sorted(os.listdir(folder_path)):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        path = os.path.join(folder_path, filename)

        image = tf.keras.utils.load_img(
            path,
            target_size=IMAGE_SIZE,
        )

        image = tf.keras.utils.img_to_array(image)

        images.append(image)
        labels.append(label)

X_test = np.array(images, dtype=np.float32)
y_true = np.array(labels, dtype=np.int32)

if len(X_test) == 0:
    raise RuntimeError("Khong co anh kiem thu!")

probabilities = model.predict(
    X_test,
    verbose=0,
).reshape(-1)

y_pred = (probabilities >= 0.5).astype(int)

print("\n===== DANH GIA CNN V3 =====")
print("Tong anh:", len(y_true))

print(
    "Accuracy:",
    round(accuracy_score(y_true, y_pred), 4),
)

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=["closed_eyes", "open_eyes"],
        zero_division=0,
    )
)

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print(
    "ROC AUC:",
    round(roc_auc_score(y_true, probabilities), 4),
)

for label, name in [(0, "CLOSED"), (1, "OPEN")]:
    group = probabilities[y_true == label]

    print(
        f"{name}: min={group.min():.6f}, "
        f"max={group.max():.6f}, "
        f"median={np.median(group):.6f}"
    )
