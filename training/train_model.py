"""Treina a CNN básica de classificação de fraturas (construída do zero), com
a mesma arquitetura e os mesmos hiperparâmetros do notebook do projeto
(Laboratório/, seções 9 a 14). Depois do treino, avalia o modelo na validação
e no teste e gera os arquivos que o app Streamlit lê:

    python training/train_model.py

Se o modelo já foi treinado no Colab, basta copiar o .keras para models/ e
rodar só a avaliação:

    python training/evaluate_model.py --model caminho/cnn_fraturas.keras
"""
import json

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from dataset_utils import HISTORY_PATH, IMG_SIZE, MODEL_PATH, MODELS_DIR, SEED, build_augmentation, load_datasets
from evaluate_model import evaluate

tf.random.set_seed(SEED)
np.random.seed(SEED)

LEARNING_RATE = 1e-3
EPOCHS = 20


def build_model():
    """CNN básica: 3 blocos Conv2D + MaxPooling, Flatten, Dense(128),
    Dropout(0.5) e saída sigmoid (probabilidade de fratura)."""
    return keras.Sequential([
        layers.Input(shape=IMG_SIZE + (3,)),
        build_augmentation(),             # ativo apenas durante o treinamento
        layers.Rescaling(1.0 / 255),      # normalização dos pixels (0-255 -> 0-1)
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid"),
    ], name="cnn_basica")


def main():
    print("Carregando datasets (treino/validação/teste)...")
    train_ds, val_ds, _, dataset_counts = load_datasets()
    print("Imagens por conjunto e classe:", json.dumps(dataset_counts, indent=2, ensure_ascii=False))

    model = build_model()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    model.summary()

    print("Treinando...")
    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[early_stopping])

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    # Sem o estado do otimizador o arquivo cai de ~134 MB para ~45 MB (cabe no GitHub).
    model.save(MODEL_PATH, include_optimizer=False)
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)
    print(f"Modelo salvo em {MODEL_PATH}")

    evaluate(MODEL_PATH)


if __name__ == "__main__":
    main()
