# Atelier 2 — Mini-projet ML + Git

Baseline ML reproductible + versioning Git + collaboration GitHub (PR, branches, tag `v0.1.0`).

> Dataset par défaut : **Iris** (scikit-learn intégré, aucun fichier externe requis).

---

## Structure

```
mlops-ml-project/
  config/train.yaml          # hyperparamètres
  src/                       # data.py, features.py, model.py
  scripts/                   # train.py, evaluate.py, git cheatsheet
  tests/test_data.py
  .github/                   # PR template, issue template
  data/                      # ⛔ ignoré par Git
  artifacts/                 # ⛔ ignoré par Git (model, metrics, plots)
```

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Exécution

```bash
python scripts/train.py          # → artifacts/model.joblib, metrics.json, confusion_matrix.png
python scripts/evaluate.py       # → artifacts/report.json
pytest -q                        # tests unitaires
```

Pour utiliser un CSV personnalisé, déposez-le dans `data/` et modifiez `config/train.yaml` (`source: "csv"`, `csv_path`, `target`).

---

## Exercice Git/GitHub

```bash
# 1. Init + commits
git init
git add .gitignore && git commit -m "chore: add .gitignore"
git add README.md requirements.txt config/ src/ scripts/ tests/ notebooks/ .github/
git commit -m "init: baseline ML project"

# 2. Push vers GitHub
git branch -M main
git remote add origin <URL>
git push -u origin main

# 3. Branches dev + feature
git checkout -b dev && git push -u origin dev
git checkout -b feature/preprocessing
git add src/features.py && git commit -m "feat: improve preprocessing"
git push -u origin feature/preprocessing
# → PR feature/preprocessing → dev, review, merge

# 4. Tag release
git checkout main && git merge dev
git tag -a v0.1.0 -m "Baseline model ready"
git push origin main --tags
```

---

## Livrables attendus

- `artifacts/metrics.json` + `artifacts/report.json`
- Capture d'écran du run `train.py`
- PR sur GitHub avec description + checklist
- Tag `v0.1.0` visible sur GitHub
