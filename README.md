<img width="1849" height="860" alt="Screenshot 2026-05-03 150526" src="https://github.com/user-attachments/assets/5646a55b-64d0-41ae-82a8-d030a1d8e2a8" />
# Cloud IDS Project

Machine-learning based cloud intrusion detection using CICIDS flow features and explainable AI output.

## Setup

```powershell
cd backend
pip install -r requirements.txt
python train_model.py
python run_server.py
```

Then open:

```text
cd frontend
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
