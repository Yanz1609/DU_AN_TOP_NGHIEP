"""
Analyze CNN V3 probability scores on the glasses_test results.
Reads glasses_test_predictions.csv created by evaluate_glasses_test.py.
Does NOT modify the model or production code.
Threshold search is exploratory only; do not present the selected threshold
as a final validated value without checking it on another independent set.
"""

import csv
import math
from pathlib import Path

CSV_PATH = Path("glasses_test_predictions.csv")


def safe_div(a, b):
    return a / b if b else 0.0


def summarize(values):
    values = sorted(values)
    n = len(values)
    return {
        "count": n,
        "min": values[0],
        "max": values[-1],
        "mean": sum(values) / n,
        "median": (
            values[n // 2] if n % 2 else
            (values[n // 2 - 1] + values[n // 2]) / 2
        ),
    }


def metrics(rows, threshold):
    # label mapping used by evaluator: closed=0, open=1.
    # Predict closed if P(open) < threshold; otherwise predict open.
    tn = fp = fn = tp = 0
    for true_label, p_open in rows:
        predicted_open = p_open >= threshold
        actual_open = true_label == "open_eyes"
        if actual_open and predicted_open:
            tp += 1
        elif actual_open and not predicted_open:
            fn += 1
        elif not actual_open and predicted_open:
            fp += 1
        else:
            tn += 1

    total = tn + fp + fn + tp
    accuracy = safe_div(tn + tp, total)
    precision_open = safe_div(tp, tp + fp)
    recall_open = safe_div(tp, tp + fn)
    f1_open = safe_div(
        2 * precision_open * recall_open,
        precision_open + recall_open
    )
    precision_closed = safe_div(tn, tn + fn)
    recall_closed = safe_div(tn, tn + fp)
    f1_closed = safe_div(
        2 * precision_closed * recall_closed,
        precision_closed + recall_closed
    )
    balanced_accuracy = (recall_open + recall_closed) / 2
    macro_f1 = (f1_open + f1_closed) / 2

    return {
        "threshold": threshold,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "accuracy": accuracy,
        "precision_open": precision_open,
        "recall_open": recall_open,
        "f1_open": f1_open,
        "precision_closed": precision_closed,
        "recall_closed": recall_closed,
        "f1_closed": f1_closed,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
    }


def main():
    if not CSV_PATH.is_file():
        raise SystemExit(
            f"Khong tim thay {CSV_PATH}. Hay chay evaluate_glasses_test.py truoc."
        )

    rows = []
    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"true_label", "p_open"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise SystemExit(
                f"CSV can co cot {sorted(required)}. "
                f"Cac cot hien tai: {reader.fieldnames}"
            )
        for row in reader:
            label = row["true_label"].strip()
            if label not in ("closed_eyes", "open_eyes"):
                continue
            p_open = float(row["p_open"])
            if not math.isfinite(p_open) or not 0 <= p_open <= 1:
                raise SystemExit(f"Xac suat khong hop le: {row}")
            rows.append((label, p_open))

    if not rows:
        raise SystemExit("CSV khong co dong du lieu hop le.")

    closed = [p for label, p in rows if label == "closed_eyes"]
    opened = [p for label, p in rows if label == "open_eyes"]
    if not closed or not opened:
        raise SystemExit("Can ca hai lop closed_eyes va open_eyes.")

    print("PHAN TICH XAC SUAT P(open_eyes)")
    print(f"Tong so anh: {len(rows)}")
    for name, vals in (("closed_eyes", closed), ("open_eyes", opened)):
        s = summarize(vals)
        print(
            f"{name:12s}: n={s['count']:2d}, min={s['min']:.6f}, "
            f"median={s['median']:.6f}, mean={s['mean']:.6f}, "
            f"max={s['max']:.6f}"
        )

    current = metrics(rows, 0.5)
    print("\nNGUONG HIEN TAI 0.500")
    print(
        f"TN={current['tn']} FP={current['fp']} "
        f"FN={current['fn']} TP={current['tp']}"
    )
    print(
        f"Accuracy={current['accuracy']:.4f}; "
        f"Balanced accuracy={current['balanced_accuracy']:.4f}; "
        f"Macro F1={current['macro_f1']:.4f}"
    )

    # Try a midpoint between each pair of adjacent unique scores, as well as
    # the existing threshold. Include margins so all-positive/all-negative
    # predictions can also be represented.
    unique_scores = sorted({p for _, p in rows})
    candidates = {0.5, 0.0, 1.0}
    if unique_scores:
        candidates.add(max(0.0, unique_scores[0] - 1e-9))
        candidates.add(min(1.0, unique_scores[-1] + 1e-9))
    for left, right in zip(unique_scores, unique_scores[1:]):
        candidates.add((left + right) / 2.0)

    evaluated = [metrics(rows, threshold) for threshold in sorted(candidates)]
    best_balanced = max(evaluated, key=lambda m: (
        m["balanced_accuracy"], m["macro_f1"], m["accuracy"]
    ))
    best_accuracy = max(evaluated, key=lambda m: (
        m["accuracy"], m["balanced_accuracy"], m["macro_f1"]
    ))

    print("\nNGUONG THAM KHAO TOT NHAT TREN CHINH BO ANH NAY")
    print(
        f"Theo balanced accuracy: threshold={best_balanced['threshold']:.6f}; "
        f"balanced_accuracy={best_balanced['balanced_accuracy']:.4f}; "
        f"accuracy={best_balanced['accuracy']:.4f}; "
        f"macro_f1={best_balanced['macro_f1']:.4f}"
    )
    print(
        f"TN={best_balanced['tn']} FP={best_balanced['fp']} "
        f"FN={best_balanced['fn']} TP={best_balanced['tp']}"
    )
    print(
        f"Theo accuracy: threshold={best_accuracy['threshold']:.6f}; "
        f"accuracy={best_accuracy['accuracy']:.4f}; "
        f"balanced_accuracy={best_accuracy['balanced_accuracy']:.4f}; "
        f"macro_f1={best_accuracy['macro_f1']:.4f}"
    )
    print(
        f"TN={best_accuracy['tn']} FP={best_accuracy['fp']} "
        f"FN={best_accuracy['fn']} TP={best_accuracy['tp']}"
    )

    if max(closed) < min(opened):
        boundary = (max(closed) + min(opened)) / 2.0
        print(
            "\nKiem tra xep hang: cac diem P(open) cua lop closed deu thap hon "
            "lop open tren bo anh nay."
        )
        print(
            f"Khoang tach lop mau: max(closed)={max(closed):.6f}, "
            f"min(open)={min(opened):.6f}, midpoint={boundary:.6f}"
        )
    elif max(opened) < min(closed):
        print(
            "\nCANH BAO: Diem P(open) co ve dao chieu so voi cach dat ten lop. "
            "Can kiem tra mapping dau ra CNN truoc khi thay threshold."
        )
        print(
            f"max(open)={max(opened):.6f}, min(closed)={min(closed):.6f}"
        )
    else:
        print(
            "\nHai lop co diem P(open) chong lan nhau; mot nguong duy nhat "
            "khong the tach hoan hao tren bo anh nay."
        )

    print(
        "\nQUAN TRONG: Nguong duoc tim tren chinh bo glasses_test de chan doan, "
        "nen ket qua toi uu co nguy co qua khop voi 40 anh nay. "
        "Chi chon nguong cuoi cung sau khi kiem tra tren mot tap validation "
        "rieng va giu mot tap test doc lap."
    )


if __name__ == "__main__":
    main()
