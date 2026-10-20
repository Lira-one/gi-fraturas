"""Constantes e utilidades de dataset/pré-processamento compartilhadas entre
o script de treino (training/train_model.py) e o app Streamlit
(streamlit_app/model_utils.py). Mantidas num único lugar para garantir que a
inferência em produção use exatamente o mesmo pipeline usado no treino.

Metodologia espelhada do notebook de referência do projeto
(Laboratório/Deteccao_Fraturas.ipynb, "Grupo GI"): classe positiva explícita
(fractured = índice 1, igual ao padrão do scikit-learn/Keras), e um conjunto
de teste real, nunca visto no treino, separado de forma estratificada a
partir de train/ (já que a pasta test/ do dataset original está vazia). A
pasta validation/ original é usada como está, para o Early Stopping.
"""
import random
from pathlib import Path

import tensorflow as tf

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "DataSet Raio-x" / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
VALIDATION_DIR = DATASET_DIR / "validation"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "fracture_cnn_mobilenetv2.keras"
METRICS_PATH = MODELS_DIR / "metrics.json"

# Classe positiva (índice 1) = "fractured", para que a saída sigmoid do
# modelo seja diretamente P(fratura) e as métricas do sklearn (que usam
# pos_label=1 por padrão) já meçam a classe de interesse clínico.
CLASS_NAMES = ["not fractured", "fractured"]
DISPLAY_NAMES = ["Sem fratura", "Fraturado"]

TEST_FRACTION = 0.15


def _list_images(class_dir: Path):
    return sorted(str(p) for p in class_dir.glob("*.jpg"))


def _stratified_holdout_split(seed: int = SEED, fraction: float = TEST_FRACTION):
    """Separa, de forma estratificada por classe, uma fração de train/ para
    servir de conjunto de teste real (nunca visto no treino nem usado para
    Early Stopping). Mesma lógica (ordenar + random.Random(seed).shuffle +
    pegar os primeiros N) do notebook de referência do projeto.
    """
    train_paths, train_labels = [], []
    test_paths, test_labels = [], []
    for label, class_name in enumerate(CLASS_NAMES):
        files = _list_images(TRAIN_DIR / class_name)
        shuffled = files[:]
        random.Random(seed).shuffle(shuffled)
        n_test = max(1, round(len(shuffled) * fraction))
        test_set = set(shuffled[:n_test])
        for f in files:
            if f in test_set:
                test_paths.append(f)
                test_labels.append(label)
            else:
                train_paths.append(f)
                train_labels.append(label)
    return (train_paths, train_labels), (test_paths, test_labels)


def _list_validation():
    paths, labels = [], []
    for label, class_name in enumerate(CLASS_NAMES):
        for f in _list_images(VALIDATION_DIR / class_name):
            paths.append(f)
            labels.append(label)
    return paths, labels


def _make_dataset(paths, labels, shuffle: bool, seed: int = SEED):
    def _load(path, label):
        raw = tf.io.read_file(path)
        image = tf.image.decode_jpeg(raw, channels=3)
        image = tf.image.resize(image, IMG_SIZE)
        return image, tf.expand_dims(tf.cast(label, tf.float32), axis=-1)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=seed)
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


def load_datasets():
    """Carrega os três conjuntos: treino (85% de train/), validação (pasta
    validation/ original, usada para Early Stopping) e teste (15% de train/,
    holdout real nunca visto durante o treino).
    """
    (train_paths, train_labels), (test_paths, test_labels) = _stratified_holdout_split()
    val_paths, val_labels = _list_validation()

    train_ds = _make_dataset(train_paths, train_labels, shuffle=True)
    val_ds = _make_dataset(val_paths, val_labels, shuffle=False)
    test_ds = _make_dataset(test_paths, test_labels, shuffle=False)

    counts = {
        "train": {
            CLASS_NAMES[0]: train_labels.count(0),
            CLASS_NAMES[1]: train_labels.count(1),
        },
        "validation": {
            CLASS_NAMES[0]: val_labels.count(0),
            CLASS_NAMES[1]: val_labels.count(1),
        },
        "test": {
            CLASS_NAMES[0]: test_labels.count(0),
            CLASS_NAMES[1]: test_labels.count(1),
        },
    }
    return train_ds, val_ds, test_ds, counts


def build_augmentation():
    """Augmentation leve, aplicada apenas no conjunto de treino (camada
    Keras: fica inativa automaticamente em evaluate()/predict()).
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.1),
    ], name="data_augmentation")
