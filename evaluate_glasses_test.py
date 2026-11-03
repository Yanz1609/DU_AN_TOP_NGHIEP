"""
Evaluate CNN V3 separately on a new glasses-only image set.

Expected folders:
data/glasses_test/
    closed_eyes/
    open_eyes/

The model output is interpreted as P(open_eyes), matching camera_stream.py.
This script evaluates CNN only; it does not use EAR, FaceMesh, or the alert logic.
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABELS = ("closed_eyes", "open_eyes")


def load_images(data_dir: Path):
    images = []
    y_true = []
    paths = []
    skipped = []

    for label in LABELS:
        folder = data_dir / label
        if not folder.is_dir():
            raise FileNotFoundError(
                f"Khong tim thay thu muc: {folder}\n"
                f"Can co ca {data_dir / 'closed_eyes'} va "
                f"{data_dir / 'open_eyes'}."
            )

        image_paths = sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        if not image_paths:
            raise RuntimeError(f"Khong co anh hop le trong: {folder}")

        for image_path in image_paths:
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                skipped.append(str(image_path))
                continue

            # Match the CNN V3 inference path: 96x96, RGB, float32.
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            image_rgb = cv2.resize(
                image_rgb, (96, 96), interpolation=cv2.INTER_AREA
            )
            images.append(image_rgb.astype(np.float32))
            y_true.append(0 if label == "closed_eyes" else 1)
            paths.append(image_path)

    if not images:
        raise RuntimeError("Khong doc duoc anh nao de danh gia.")

    return np.stack(images), np.asarray(y_true, dtype=np.int32), paths, skipped


def safe_divide(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def class_metrics(cm, class_index):
    tp = int(cm[class_index, class_index])
    fp = int(cm[1 - class_index, class_index])
    fn = int(cm[class_index, 1 - class_index])

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    return precision, recall, f1


def binary_auc(labels_positive, scores):
    """AUC via average ranks; labels_positive=True denotes positive class."""
    labels_positive = np.asarray(labels_positive, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(np.sum(labels_positive))
    negatives = int(len(labels_positive) - positives)

    if positives == 0 or negatives == 0:
        return None

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)

    i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        # Ranks are one-based; tied scores get their average rank.
        average_rank = ((i + 1) + j) / 2.0
        ranks[order[i:j]] = average_rank
        i = j

    rank_sum_positive = float(np.sum(ranks[labels_positive]))
    return (
        rank_sum_positive - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def main():
    parser = argparse.ArgumentParser(
        description="Danh gia rieng CNN V3 tren bo anh deo kinh."
    )
    parser.add_argument(
        "--data-dir", default="data/glasses_test",
        help="Thu muc chua closed_eyes va open_eyes."
    )
    parser.add_argument(
        "--model", default="models/eye_cnn_v3.keras",
        help="Duong dan mo hinh CNN V3."
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Nguong P(open_eyes); mac dinh 0.5."
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    model_path = Path(args.model)

    if not model_path.is_file():
        print(f"LOI: Khong tim thay mo hinh: {model_path}")
        sys.exit(1)
    if not 0.0 < args.threshold < 1.0:
        print("LOI: --threshold phai nam trong khoang 0 va 1.")
        sys.exit(1)

    try:
        x, y_true, image_paths, skipped = load_images(data_dir)
        print(f"So anh doc duoc: {len(x)}")
        print("Dang nap CNN V3...")
        model = tf.keras.models.load_model(str(model_path))
        print("Dang du doan CNN (khong dung EAR)...")
        raw_predictions = model(x, training=False).numpy().reshape(-1)

        if len(raw_predictions) != len(y_true):
            raise RuntimeError(
                "So du doan khong khop voi so anh dau vao."
            )
        if not np.all(np.isfinite(raw_predictions)):
            raise RuntimeError("Mo hinh tao ra du doan NaN/Inf.")
        if np.min(raw_predictions) < 0.0 or np.max(raw_predictions) > 1.0:
            raise RuntimeError(
                "Dau ra CNN nam ngoai [0, 1]. "
                "Kiem tra lai mo hinh co tra ve xac suat sigmoid khong."
            )

        # Existing camera_stream.py treats model output as probability of open eyes.
        p_open = raw_predictions
        y_pred = np.where(p_open < args.threshold, 0, 1).astype(np.int32)

        # Rows = actual [closed, open]; columns = predicted [closed, open].
        cm = np.zeros((2, 2), dtype=np.int64)
        for actual, predicted in zip(y_true, y_pred):
            cm[int(actual), int(predicted)] += 1

        accuracy = float(np.mean(y_true == y_pred))
        closed_precision, closed_recall, closed_f1 = class_metrics(cm, 0)
        open_precision, open_recall, open_f1 = class_metrics(cm, 1)
        macro_precision = (closed_precision + open_precision) / 2.0
        macro_recall = (closed_recall + open_recall) / 2.0
        macro_f1 = (closed_f1 + open_f1) / 2.0
        auc = binary_auc(y_true == 0, 1.0 - p_open)

        dataset_tag = data_dir.name
        threshold_tag = f"{args.threshold:.3f}".replace(".", "")

        output_csv = Path(
            f"{dataset_tag}_predictions_t{threshold_tag}.csv"
        )

        report_path = Path(
            f"{dataset_tag}_metrics_t{threshold_tag}.txt"
        )
        with output_csv.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([
                "image_path", "true_label", "p_open", "p_closed",
                "predicted_label", "correct"
            ])
            for path, actual, po, predicted in zip(
                image_paths, y_true, p_open, y_pred
            ):
                writer.writerow([
                    str(path),
                    LABELS[int(actual)],
                    f"{float(po):.6f}",
                    f"{1.0 - float(po):.6f}",
                    LABELS[int(predicted)],
                    "YES" if actual == predicted else "NO",
                ])

        lines = [
            "CNN V3 - KET QUA KIEM THU RIENG KHI DEO KINH",
            f"Model: {model_path}",
            f"Dataset: {data_dir}",
            f"Threshold P(open_eyes): {args.threshold:.3f}",
            f"Total images: {len(y_true)}",
            f"Closed eyes: {int(np.sum(y_true == 0))}",
            f"Open eyes: {int(np.sum(y_true == 1))}",
            "",
            "Confusion matrix (rows=true, columns=predicted)",
            "labels = [closed_eyes, open_eyes]",
            f"[[{cm[0,0]}, {cm[0,1]}], [{cm[1,0]}, {cm[1,1]}]]",
            "",
            f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)",
            f"Closed eyes - Precision: {closed_precision:.4f}",
            f"Closed eyes - Recall:    {closed_recall:.4f}",
            f"Closed eyes - F1-score:  {closed_f1:.4f}",
            f"Open eyes   - Precision: {open_precision:.4f}",
            f"Open eyes   - Recall:    {open_recall:.4f}",
            f"Open eyes   - F1-score:  {open_f1:.4f}",
            f"Macro Precision: {macro_precision:.4f}",
            f"Macro Recall:    {macro_recall:.4f}",
            f"Macro F1-score:  {macro_f1:.4f}",
            f"ROC AUC (positive class=closed_eyes): "
            f"{auc:.4f}" if auc is not None else
            "ROC AUC: N/A (can requires both classes)",
            "",
            f"CSV predictions: {output_csv.resolve()}",
        ]
        if skipped:
            lines += [
                f"Skipped unreadable files: {len(skipped)}",
                *[f"  - {p}" for p in skipped],
            ]

        report = "\n".join(lines)
        print("\n" + report)
        report_path.write_text(
            report + "\n",
            encoding="utf-8"
        )

        print(f"\nDa luu bao cao: {report_path}")

        if len(y_true) < 40:
            print(
                "\nLUU Y: So anh it hon 40; kiem tra lai du lieu neu ban "
                "mong doi 20 anh moi lop."
            )
        print(
            "\nLuu y: Ket qua nay danh gia CNN tren anh da thu, khong phai "
            "ket qua canh bao tong hop CNN + EAR va khong bao dam kha nang "
            "tong quat tren moi nguoi/dieu kien."
        )

    except Exception as exc:
        print(f"\nLOI KHI DANH GIA: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
