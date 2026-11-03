
import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# ===== CAU HINH =====
MODEL_PATH = "models/eye_cnn.keras"
TEST_DIR = "data/test"

# Thu tu nhan giong khi huan luyen
CLASSES = ["closed_eyes", "open_eyes"]

# ===== TAI MO HINH =====
model = tf.keras.models.load_model(MODEL_PATH)

images = []
true_labels = []

# ===== DOC ANH KIEM THU =====
for label, class_name in enumerate(CLASSES):
    folder = os.path.join(TEST_DIR, class_name)

    for filename in sorted(os.listdir(folder)):
        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            continue

        path = os.path.join(folder, filename)
        image = cv2.imread(path)

        if image is None:
            print("Khong doc duoc:", path)
            continue

        image = cv2.resize(image, (96, 96))
        image = cv2.cvtColor(
            image, cv2.COLOR_BGR2RGB
        )

        images.append(image)
        true_labels.append(label)

if not images:
    raise RuntimeError("Khong tim thay anh kiem thu!")

X_test = np.array(images, dtype=np.float32)
y_true = np.array(true_labels)

print("Tong so anh kiem thu:", len(X_test))

# Model da co lop Rescaling nen khong chia 255
probabilities = model.predict(
    X_test, verbose=0
).flatten()

# closed_eyes = 0, open_eyes = 1
y_pred = (probabilities >= 0.5).astype(int)

# ===== DANH GIA =====
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(
    y_true, y_pred, zero_division=0
)
recall = recall_score(
    y_true, y_pred, zero_division=0
)
f1 = f1_score(
    y_true, y_pred, zero_division=0
)

print("\n===== KET QUA CNN =====")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(
    y_true,
    y_pred,
    target_names=CLASSES,
    zero_division=0
))

# ===== MA TRAN NHAM LAN =====
cm = confusion_matrix(
    y_true, y_pred, labels=[0, 1]
)

print("\nConfusion Matrix:")
print(cm)

os.makedirs("results", exist_ok=True)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Closed", "Open"]
)

display.plot(cmap="Blues", values_format="d")
plt.title("CNN - Confusion Matrix")
plt.tight_layout()
plt.savefig(
    "results/confusion_matrix.png",
    dpi=200
)
plt.show()

print("\nDa luu: results/confusion_matrix.png")
