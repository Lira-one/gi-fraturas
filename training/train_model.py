"""Treina o classificador binário de fraturas ósseas via Transfer Learning
(MobileNetV2 com base congelada), seguindo a mesma arquitetura e metodologia
da seção "Transfer Learning" do notebook de referência do projeto
(Laboratório/Deteccao_Fraturas.ipynb). Salva o modelo + métricas de avaliação
para o app Streamlit consumir. Roda uma única vez (ou sob demanda), fora do
Streamlit:

    python training/train_model.py

Não treinamos aqui a CNN "do zero" (também presente no notebook de
referência): em CPU, sem GPU, o custo de treinar todas as camadas
convolucionais por até 20 épocas é proibitivo. O Transfer Learning (apenas a
cabeça de classificação é treinada) já obteve bom recall para a classe
Fratura em execução anterior e é viável em CPU.
"""
import json
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from dataset_utils import (
    CLASS_NAMES,
    DISPLAY_NAMES,
    IMG_SIZE,
    MODEL_PATH,
    MODELS_DIR,
    METRICS_PATH,
    SEED,
    build_augmentation,
    load_datasets,
)

tf.random.set_seed(SEED)
np.random.seed(SEED)

LEARNING_RATE = 1e-3
EPOCHS = 10
THRESHOLD = 0.5


def build_model():
    data_augmentation = build_augmentation()
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet"
    )
    base_model.trainable = False  # congela a base pré-treinada no ImageNet

    inputs = keras.Input(shape=IMG_SIZE + (3,))
    x = data_augmentation(inputs)  # só atua no treino (training=True)
    x = layers.Rescaling(1.0 / 127.5, offset=-1)(x)  # pixels 0-255 -> [-1, 1]
    x = base_model(x, training=False)  # base congelada (BatchNorm em modo inferência)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs, name="mobilenetv2_transfer")


def main():
    print("Carregando datasets (treino/validação/teste)...")
    train_ds, val_ds, test_ds, dataset_counts = load_datasets()
    print("Imagens por conjunto e classe:", json.dumps(dataset_counts, indent=2, ensure_ascii=False))

    model = build_model()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")],
    )

    print("Treinando...")
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[early_stopping])

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Modelo salvo em {MODEL_PATH}")

    print("Avaliando no conjunto de teste (holdout real, nunca visto no treino)...")
    y_true, y_prob = [], []
    for images, labels in test_ds:
        probs = model(images, training=False).numpy().ravel()
        y_true.append(labels.numpy().ravel())
        y_prob.append(probs)
    y_true = np.concatenate(y_true).astype(int)
    y_prob = np.concatenate(y_prob)
    y_pred = (y_prob >= THRESHOLD).astype(int)

    # pos_label=1 é o padrão do sklearn e, com CLASS_NAMES = ["not fractured",
    # "fractured"], já corresponde à classe "fractured" (interesse clínico).
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_prob),
        "confusion_matrix": cm.tolist(),
        "confusion_counts": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "class_names": CLASS_NAMES,
        "display_names": DISPLAY_NAMES,
        "threshold": THRESHOLD,
        "test_size": int(len(y_true)),
        "train_history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
        "dataset_counts": dataset_counts,
        "trained_at": datetime.now().isoformat(),
        "model_architecture": "MobileNetV2 (Transfer Learning, base congelada)",
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Métricas salvas em {METRICS_PATH}")
    print(f"Acurácia: {metrics['accuracy']:.2%}")
    print(
        f"[Fratura] Precisão: {metrics['precision']:.2%} | "
        f"Recall: {metrics['recall']:.2%} | F1: {metrics['f1_score']:.2%} | AUC: {metrics['auc']:.4f}"
    )
    print(f"TN={tn} FP={fp} FN={fn} TP={tp}")


if __name__ == "__main__":
    main()
