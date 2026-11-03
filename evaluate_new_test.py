
import os
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# Cau hinh
MODEL_PATH = "models/eye_cnn_v3.keras"
TEST_DIR = "data/new_test"
IMAGE_SIZE = (96, 96)
THRESHOLD = 0.5

CLASS_NAMES = ["closed_eyes", "open_eyes"]


def main():
    print("===== DANH GIA CNN V3 - DU LIEU MOI =====")

    model = tf.keras.models.load_model(MODEL_PATH)

    images = []
    labels = []

    for label, class_name in enumerate(CLASS_NAMES):
        folder = os.path.join(TEST_DIR, class_name)

        if not os.path.isdir(folder):
            raise FileNotFoundError(folder)

        for filename in sorted(os.listdir(folder)):
            if not filename.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):
                continue

            path = os.path.join(folder, filename)

            image = tf.keras.utils.load_img(
                path,
                target_size=IMAGE_SIZE
            )

            image_array = tf.keras.utils.img_to_array(
                image
            )

            images.append(image_array)
            labels.append(label)

    if not images:
        print("Khong tim thay anh kiem thu!")
        return

    x_test = np.array(images, dtype=np.float32)
    y_true = np.array(labels, dtype=np.int32)

    # CNN V3 da co lop Rescaling ben trong
    # Nen khong chia them cho 255
    probabilities = model.predict(
        x_test,
        verbose=0
    ).reshape(-1)

    y_pred = (
        probabilities >= THRESHOLD
    ).astype(int)

    print("Tong anh:", len(y_true))

    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=CLASS_NAMES,
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(
        confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1]
        )
    )

    if len(np.unique(y_true)) == 2:
        auc = roc_auc_score(
            y_true,
            probabilities
        )
        print(f"ROC AUC: {auc:.4f}")

    for label, class_name in enumerate(CLASS_NAMES):
        values = probabilities[y_true == label]

        print(
            f"{class_name}: "
            f"min={values.min():.6f}, "
            f"max={values.max():.6f}, "
            f"median={np.median(values):.6f}"
        )


if __name__ == "__main__":
    main()
