from __future__ import annotations

import sys
from pathlib import Path

# Permet d'importer le package `src/` sans installation
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import yaml
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from src.data import load_data
from src.features import build_preprocessor
from src.model import build_model


def load_cfg(path: str = "config/train.yaml") -> dict:
    return yaml.safe_load(open(path, "r", encoding="utf-8"))


def set_seeds(seed: int) -> None:
    np.random.seed(seed)


def save_confusion_matrix(y_true, y_pred, out_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.title("Confusion matrix")
    plt.xlabel("Pred")
    plt.ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    cfg = load_cfg()
    art_dir = Path(cfg.get("artifacts_dir", "artifacts"))
    art_dir.mkdir(parents=True, exist_ok=True)

    split = cfg.get("split", {})
    seed = int(split.get("random_state", 42))
    set_seeds(seed)

    bundle = load_data(cfg)
    X, y = bundle.X, bundle.y

    stratify = y if bool(split.get("stratify", True)) else None
    Xtr, Xte, ytr, yte = train_test_split(
        X, y,
        test_size=float(split.get("test_size", 0.2)),
        random_state=seed,
        stratify=stratify,
    )

    pre = build_preprocessor(Xtr)
    model = build_model(cfg)

    pipe = Pipeline(steps=[
        ("preprocess", pre),
        ("model", model),
    ])

    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)

    # Metrics (macro F1 pour multi-classe)
    acc = float(accuracy_score(yte, pred))
    f1 = float(f1_score(yte, pred, average="macro"))

    joblib.dump(pipe, art_dir / "model.joblib")

    json.dump(
        {"accuracy": acc, "f1_macro": f1},
        open(art_dir / "metrics.json", "w", encoding="utf-8"),
        indent=2,
    )

    save_confusion_matrix(yte, pred, art_dir / "confusion_matrix.png")

    run_info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": cfg,
        "data_meta": bundle.meta,
        "versions": {
            "python": __import__("sys").version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
        },
    }
    json.dump(
        run_info,
        open(art_dir / "run_info.json", "w", encoding="utf-8"),
        indent=2,
    )

    print("Train OK")
    print(f" - accuracy   : {acc:.4f}")
    print(f" - f1_macro   : {f1:.4f}")
    print(f" - artifacts  : {art_dir.resolve()}")


if __name__ == "__main__":
    main()
