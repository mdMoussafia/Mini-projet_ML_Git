from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import pandas as pd
from sklearn.datasets import load_iris


@dataclass
class DataBundle:
    X: pd.DataFrame
    y: pd.Series
    problem_type: str  # "classification"
    label_name: str
    meta: Dict[str, Any]


def load_data(cfg: Dict[str, Any]) -> DataBundle:
    '''
    Charge les données selon la config.
    - iris: dataset intégré scikit-learn
    - csv : fichier data/dataset.csv (ou autre chemin) + colonne target
    '''
    data_cfg = cfg.get("data", {})
    source = (data_cfg.get("source") or "iris").lower()

    if source == "iris":
        X, y = load_iris(return_X_y=True, as_frame=True)
        return DataBundle(
            X=X,
            y=y,
            problem_type="classification",
            label_name="iris_class",
            meta={"source": "iris", "n_samples": int(X.shape[0]), "n_features": int(X.shape[1])},
        )

    if source == "csv":
        path = data_cfg.get("csv_path", "data/dataset.csv")
        target = data_cfg.get("target", "target")
        df = pd.read_csv(path)

        if target not in df.columns:
            raise ValueError(f"Colonne target '{target}' introuvable dans {path}. Colonnes: {list(df.columns)}")

        y = df[target]
        X = df.drop(columns=[target])

        return DataBundle(
            X=X,
            y=y,
            problem_type="classification",
            label_name=target,
            meta={"source": "csv", "path": path, "n_samples": int(X.shape[0]), "n_features": int(X.shape[1])},
        )

    raise ValueError(f"data.source non supporté: {source!r} (attendu: 'iris' ou 'csv')")
