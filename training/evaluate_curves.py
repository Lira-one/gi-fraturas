"""Gera as curvas de avaliação do notebook de referência
(Laboratório/Deteccao_Fraturas.ipynb, seções 24, 25 e 32) a partir do modelo
já treinado, sobre o MESMO conjunto de teste (holdout) usado em
train_model.py, e salva em models/evaluation_curves.json para o painel do
Gerente consumir sem precisar rodar inferência no Streamlit:

    python training/evaluate_curves.py

- Curva ROC e AUC (seção 24)
- Curva Precision-Recall e Average Precision (seção 25)
- Efeito do threshold (seção 25, "Efeito do threshold")
- Curva de calibração / reliability diagram (seção 32)
"""
import json

import numpy as np
import tensorflow as tf
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from dataset_utils import MODEL_PATH, MODELS_DIR, load_datasets

CURVES_PATH = MODELS_DIR / "evaluation_curves.json"
THRESHOLDS_TABLE = [0.3, 0.4, 0.5, 0.6, 0.7]
MAX_CURVE_POINTS = 200


def _downsample(*arrays):
    """Reduz as curvas a no máximo MAX_CURVE_POINTS pontos (o JSON fica leve
    e o gráfico continua idêntico ao olho)."""
    n = len(arrays[0])
    if n <= MAX_CURVE_POINTS:
        return [a.tolist() for a in arrays]
    idx = np.unique(np.linspace(0, n - 1, MAX_CURVE_POINTS).astype(int))
    return [a[idx].tolist() for a in arrays]


def main():
    print("Carregando modelo e conjunto de teste...")
    model = tf.keras.models.load_model(MODEL_PATH)
    _, _, test_ds, _ = load_datasets()

    y_true, y_prob = [], []
    for images, labels in test_ds:
        y_prob.append(model(images, training=False).numpy().ravel())
        y_true.append(labels.numpy().ravel())
    y_true = np.concatenate(y_true).astype(int)
    y_prob = np.concatenate(y_prob)
    print(f"{len(y_true)} imagens de teste avaliadas.")

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")

    threshold_rows = []
    for t in THRESHOLDS_TABLE:
        y_hat = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()
        threshold_rows.append({
            "threshold": t,
            "precision": float(precision_score(y_true, y_hat, zero_division=0)),
            "recall": float(recall_score(y_true, y_hat, zero_division=0)),
            "f1": float(f1_score(y_true, y_hat, zero_division=0)),
            "fn": int(fn),
            "fp": int(fp),
        })

    roc_x, roc_y = _downsample(fpr, tpr)
    pr_x, pr_y = _downsample(recall, precision)
    curves = {
        "roc": {"fpr": roc_x, "tpr": roc_y, "auc": float(roc_auc_score(y_true, y_prob))},
        "pr": {
            "recall": pr_x,
            "precision": pr_y,
            "average_precision": float(average_precision_score(y_true, y_prob)),
            "baseline": float(y_true.mean()),
        },
        "calibration": {"mean_predicted": mean_pred.tolist(), "fraction_positive": frac_pos.tolist()},
        "threshold_table": threshold_rows,
        "test_size": int(len(y_true)),
    }

    with open(CURVES_PATH, "w", encoding="utf-8") as f:
        json.dump(curves, f, indent=2, ensure_ascii=False)
    print(f"Curvas salvas em {CURVES_PATH}")
    print(f"AUC = {curves['roc']['auc']:.4f} | AP = {curves['pr']['average_precision']:.4f}")


if __name__ == "__main__":
    main()
