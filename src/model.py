from __future__ import annotations

from typing import Dict, Any

from sklearn.linear_model import LogisticRegression


def build_model(cfg: Dict[str, Any]) -> LogisticRegression:
    model_cfg = cfg.get("model", {})
    name = (model_cfg.get("name") or "logistic_regression").lower()

    if name != "logistic_regression":
        raise ValueError(f"Modèle non supporté dans ce baseline: {name!r}")

    return LogisticRegression(
        max_iter=int(model_cfg.get("max_iter", 2000)),
        class_weight=model_cfg.get("class_weight", None),
        n_jobs=None,
    )
