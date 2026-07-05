# Atelier 2 — Exercice d'application (solution) : Mini-projet ML + Git

Ce dépôt correspond à un **exercice unique** qui regroupe :
1) construction d'un **baseline ML** reproductible (scripts, config, artefacts),
2) **versioning Git** (commits propres, branches main/dev/feature),
3) **collaboration GitHub** via PR (template fourni),
4) **tag/release** d'une baseline (`v0.1.0`).

> Remarque : un dossier `.git` (historique Git) n'est jamais inclus dans un ZIP.  
> Vous trouverez ci-dessous les **commandes** pour reconstruire l'historique demandé dans l'exercice.

---

## 1) Structure

```
mlops-ml-project/
  README.md
  requirements.txt
  .gitignore
  config/
    train.yaml
  src/
    __init__.py
    data.py
    features.py
    model.py
  scripts/
    train.py
    evaluate.py
    git_workflow_cheatsheet.txt
  notebooks/
    (vide - optionnel)
  data/                 # NON versionné par Git
  artifacts/            # NON versionné par Git (sorties)
  tests/
    test_data.py
  .github/
    pull_request_template.md
    ISSUE_TEMPLATE/
      feature.md
```

---

## 2) Installation (local)

### Option A — environnement virtuel (recommandé)
**Windows PowerShell**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Linux/macOS**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 3) Exécuter le baseline ML

Par défaut, la config utilise **Iris** (dataset intégré scikit-learn), donc aucun fichier externe n'est nécessaire.

### Train
```bash
python scripts/train.py
```

### Vérifier les artefacts
- `artifacts/model.joblib`
- `artifacts/metrics.json`
- `artifacts/confusion_matrix.png`
- `artifacts/run_info.json`

### Evaluate
```bash
python scripts/evaluate.py
```

Produit : `artifacts/report.json`

---

## 4) Tests
```bash
pytest -q
```

---

## 5) Utiliser un dataset CSV (option)

1) Déposez votre dataset dans `data/dataset.csv`
2) Modifiez `config/train.yaml` :

```yaml
data:
  source: "csv"
  csv_path: "data/dataset.csv"
  target: "Exited"
```

> **Important** : `data/` et `artifacts/` sont ignorés par Git (voir `.gitignore`).

---

## 6) Exercice Git/GitHub (pas-à-pas)

### 6.1 Init + commits propres
```bash
git init

# 1) .gitignore en premier
git add .gitignore
git commit -m "chore: add .gitignore for ML project"

# 2) Baseline (code/config/docs)
git add README.md requirements.txt config/ src/ scripts/ tests/ notebooks/ .github/
git commit -m "init: baseline ML project (train/eval/tests)"
```

### 6.2 Lier à GitHub + push
Créez un dépôt GitHub vide, puis :
```bash
git branch -M main
git remote add origin <URL_DU_DEPOT>
git push -u origin main
```

### 6.3 Workflow branches main/dev/feature (feature/preprocessing)
```bash
git checkout -b dev
git push -u origin dev

git checkout -b feature/preprocessing
# Exemple de changement : améliorer src/features.py
git add src/features.py
git commit -m "feat: improve preprocessing pipeline"
git push -u origin feature/preprocessing
```

Ensuite :
- ouvrir une PR **feature/preprocessing -> dev**
- review binôme
- merge dans `dev`

### 6.4 Tagger une baseline (release)
```bash
git checkout main
git merge dev
git tag -a v0.1.0 -m "Baseline model ready"
git push origin main
git push origin --tags
```

---

## 7) Aide-mémoire
Voir `scripts/git_workflow_cheatsheet.txt`.

---

## 8) Sorties attendues (preuves)
- `artifacts/metrics.json` + `artifacts/report.json`
- capture d'écran du run de `train.py`
- PR sur GitHub (avec description + checklist)
- tag `v0.1.0` visible sur GitHub
