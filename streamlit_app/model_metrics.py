"""Leitura dos artefatos de avaliação gerados fora do Streamlit:

- models/metrics.json            (training/train_model.py)
- models/evaluation_curves.json  (training/evaluate_curves.py)
"""
import json
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = PROJECT_ROOT / "models" / "metrics.json"
CURVES_PATH = PROJECT_ROOT / "models" / "evaluation_curves.json"


def _read_json(path: Path, mtime: float):
    # mtime entra na chave do cache: re-treinar o modelo invalida o cache.
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


_cached_read = st.cache_data(show_spinner=False)(_read_json)


def _load(path: Path):
    if not path.exists():
        return None
    return _cached_read(path, path.stat().st_mtime)


def load_metrics():
    return _load(METRICS_PATH)


def load_curves():
    return _load(CURVES_PATH)
