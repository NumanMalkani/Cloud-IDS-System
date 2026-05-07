# Cloud IDS Project

Machine-learning based cloud intrusion detection using CICIDS flow features and explainable AI output.

## Setup

```powershell
pip install -r requirements.txt
python train_model.py
python run_server.py
```

Then open:

```text
C:\Users\sahuk\OneDrive\Documents\6th SEM\PROJECTS\Cloud_IDS_Project\frontend\index.html
```

## API

- `GET /features` returns model feature names, default median values, and model accuracy.
- `POST /predict` accepts feature values as JSON and returns the prediction, probabilities, and explanation.
- `GET /confusion` returns the confusion matrix and classification metrics.

## Current Model

The backend trains a binary Random Forest classifier:

- `BENIGN` becomes `Normal`
- every other label becomes `Attack`

The explanation table uses SHAP values when available and falls back to feature-importance based contributions if SHAP cannot run.
