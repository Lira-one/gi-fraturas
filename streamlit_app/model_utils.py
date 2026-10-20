"""Carregamento do modelo treinado e inferência sobre imagens enviadas pelo
radiologista. Reusa as constantes de training/dataset_utils.py para garantir
paridade exata entre treino e inferência.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "training") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "training"))

import streamlit as st
import tensorflow as tf

from dataset_utils import IMG_SIZE, MODEL_PATH

THRESHOLD = 0.5


@st.cache_resource(show_spinner="Carregando modelo...")
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def model_is_available() -> bool:
    return MODEL_PATH.exists()


def prepare_image_array(pil_image):
    """Redimensiona a imagem para IMG_SIZE e devolve um lote (1, H, W, 3) com
    os pixels originais (0-255) — a normalização acontece dentro do próprio
    modelo salvo (camada Rescaling), então não pré-processamos aqui. Usado
    tanto pela predição quanto pelo Grad-CAM (gradcam.py), para garantir que
    os dois vejam exatamente a mesma imagem de entrada.
    """
    img = pil_image.convert("RGB").resize(IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img)
    return tf.expand_dims(arr, axis=0)


def predict_from_array(arr) -> dict:
    """Mapeamento de classes (igual ao treino, CLASS_NAMES explícito):
    0 = not fractured, 1 = fractured.
    """
    model = load_model()
    prob_fractured = float(model.predict(arr, verbose=0)[0][0])
    is_fractured = prob_fractured >= THRESHOLD
    confidence = prob_fractured if is_fractured else (1 - prob_fractured)

    return {
        "label": "Fraturado" if is_fractured else "Não Fraturado",
        "is_fractured": is_fractured,
        "confidence": confidence,
        "raw_probability_fractured": prob_fractured,
    }


def predict_image(pil_image) -> dict:
    """Recebe uma PIL.Image e retorna label + confiança da predição."""
    return predict_from_array(prepare_image_array(pil_image))
