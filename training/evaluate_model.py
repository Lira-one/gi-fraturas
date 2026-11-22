"""Avalia o modelo salvo na validação e no teste e gera os arquivos lidos pelo
painel do Gerente no app:

- models/metrics.json            métricas da validação e do teste
- models/evaluation_curves.json  curvas ROC, Precision-Recall, calibração e
                                 efeito do threshold (seções 24, 25 e 32 do notebook)

Uso:
    python training/evaluate_model.py                       # avalia models/cnn_fraturas.keras
    python training/evaluate_model.py --model outro.keras   # ex.: modelo baixado do Colab

Quando --model aponta para outro arquivo, ele é copiado para
models/cnn_fraturas.keras sem o estado do otimizador (o .keras salvo no Colab
tem ~134 MB, acima do limite de 100 MB do GitHub; sem o otimizador fica ~45 MB).
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from dataset_utils import (
    CLASS_NAMES,
    CURVES_PATH,
    DISPLAY_NAMES,
    HISTORY_PATH,
    METRICS_PATH,
    MODEL_PATH,
    load_datasets,
)

THRESHOLD = 0.5
THRESHOLDS_TABLE = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MAX_CURVE_POINTS = 200
MODEL_ARCHITECTURE = "CNN básica treinada do zero (3 blocos Conv2D + Dense)"


def _predict(model, ds):
    y_true, y_prob = [], []
    for images, labels in ds:
        y_prob.append(model(images, training=False).numpy().ravel())
        y_true.append(labels.numpy().ravel())
    return np.concatenate(y_true).astype(int), np.concatenate(y_prob)


def _metrics(y_true, y_prob):
    y_pred = (y_prob >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "auc": float(roc_auc_score(y_true, y_prob)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "confusion_counts": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "size": int(len(y_true)),
    }


def _downsample(*arrays):
    n = len(arrays[0])
    if n <= MAX_CURVE_POINTS:
        return [a.tolist() for a in arrays]
    idx = np.unique(np.linspace(0, n - 1, MAX_CURVE_POINTS).astype(int))
    return [a[idx].tolist() for a in arrays]


def _curves(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    roc_x, roc_y = _downsample(fpr, tpr)
    pr_x, pr_y = _downsample(recall, precision)
    return {
        "roc": {"fpr": roc_x, "tpr": roc_y, "auc": float(roc_auc_score(y_true, y_prob))},
        "pr": {
            "recall": pr_x,
            "precision": pr_y,
            "average_precision": float(average_precision_score(y_true, y_prob)),
            "baseline": float(y_true.mean()),
        },
    }


def _threshold_table(y_true, y_prob):
    rows = []
    for t in THRESHOLDS_TABLE:
        y_hat = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()
        rows.append({
            "threshold": t,
            "precision": float(precision_score(y_true, y_hat, zero_division=0)),
            "recall": float(recall_score(y_true, y_hat, zero_division=0)),
            "f1": float(f1_score(y_true, y_hat, zero_division=0)),
            "fn": int(fn),
            "fp": int(fp),
        })
    return rows


def evaluate(model_path: Path):
    model_path = Path(model_path)
    print(f"Carregando modelo: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    if model_path.resolve() != MODEL_PATH.resolve():
        model.save(MODEL_PATH, include_optimizer=False)
        print(f"Cópia sem otimizador salva em {MODEL_PATH} ({MODEL_PATH.stat().st_size / 1e6:.1f} MB)")

    _, val_ds, test_ds, dataset_counts = load_datasets()
    y_val, p_val = _predict(model, val_ds)
    y_test, p_test = _predict(model, test_ds)
    val_metrics = _metrics(y_val, p_val)
    test_metrics = _metrics(y_test, p_test)

    history = None
    if HISTORY_PATH.exists():
        with open(HISTORY_PATH, encoding="utf-8") as f:
            history = json.load(f)

    metrics = {
        **{k: v for k, v in test_metrics.items() if k != "size"},
        "test_size": test_metrics["size"],
        "validation": val_metrics,
        "class_names": CLASS_NAMES,
        "display_names": DISPLAY_NAMES,
        "threshold": THRESHOLD,
        "dataset_counts": dataset_counts,
        "trained_at": datetime.fromtimestamp(MODEL_PATH.stat().st_mtime).isoformat(),
        "model_architecture": MODEL_ARCHITECTURE,
        "model_params": int(model.count_params()),
    }
    if history:
        metrics["train_history"] = history

    frac_pos, mean_pred = calibration_curve(y_test, p_test, n_bins=10, strategy="uniform")
    curves = {
        "test": _curves(y_test, p_test),
        "validation": _curves(y_val, p_val),
        "calibration": {"mean_predicted": mean_pred.tolist(), "fraction_positive": frac_pos.tolist()},
        "threshold_table": _threshold_table(y_test, p_test),
        "test_size": int(len(y_test)),
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(CURVES_PATH, "w", encoding="utf-8") as f:
        json.dump(curves, f, indent=2, ensure_ascii=False)

    for nome, m in [("Validação", val_metrics), ("Teste", test_metrics)]:
        c = m["confusion_counts"]
        print(f"[{nome}] Accuracy {m['accuracy']:.2%} | Precision {m['precision']:.2%} | "
              f"Recall {m['recall']:.2%} | F1 {m['f1_score']:.2%} | AUC {m['auc']:.4f} | "
              f"TN={c['tn']} FP={c['fp']} FN={c['fn']} TP={c['tp']}")
    print(f"Arquivos gerados: {METRICS_PATH.name}, {CURVES_PATH.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(MODEL_PATH), help="arquivo .keras a avaliar")
    evaluate(Path(parser.parse_args().model))
