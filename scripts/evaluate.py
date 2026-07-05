from __future__ import annotations

import sys
from pathlib import Path

# Permet d'importer le package `src/` sans installation
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
from pathlib import Path

import joblib
import yaml
from sklearn.metrics import classification_report

from src.data import load_data


def load_cfg(path: str = "config/train.yaml") -> dict:
    return yaml.safe_load(open(path, "r", encoding="utf-8"))


def main() -> None:
    cfg = load_cfg()
    art_dir = Path(cfg.get("artifacts_dir", "artifacts"))
    model_path = art_dir / "model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Modèle introuvable: {model_path}. Lancez d'abord `python scripts/train.py`."
        )

    model = joblib.load(model_path)
    bundle = load_data(cfg)

    pred = model.predict(bundle.X)
    report = classification_report(bundle.y, pred, output_dict=True)

    art_dir.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(art_dir / "report.json", "w", encoding="utf-8"), indent=2)

    print("Evaluate OK - report saved to artifacts/report.json")


if __name__ == "__main__":
    main()
