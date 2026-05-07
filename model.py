from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

try:
    import shap
except Exception:  # pragma: no cover - SHAP is optional at runtime
    shap = None


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATASET_DIR = PROJECT_DIR / "dataset"
MODEL_PATH = BASE_DIR / "model.pkl"
ARTIFACT_VERSION = 2

FEATURES = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Mean",
    "Packet Length Mean",
    "Packet Length Std",
    "SYN Flag Count",
    "ACK Flag Count",
    "PSH Flag Count",
    "Init_Win_bytes_forward",
]

LABEL_COL = "Label"


def _clean_columns(df):
    df = df.copy()
    df.columns = df.columns.str.strip()
    return df


def _read_training_rows(max_rows_per_file=25000):
    files = sorted(DATASET_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {DATASET_DIR}")

    frames = []
    usecols = FEATURES + [LABEL_COL]

    for csv_file in files:
        file_frames = []
        for chunk in pd.read_csv(csv_file, chunksize=50000, low_memory=False, encoding_errors="replace"):
            chunk = _clean_columns(chunk)
            missing = [col for col in usecols if col not in chunk.columns]
            if missing:
                raise ValueError(f"{csv_file.name} is missing columns: {missing}")

            chunk = chunk[usecols]
            chunk[LABEL_COL] = chunk[LABEL_COL].astype(str).str.strip()

            benign = chunk[chunk[LABEL_COL].eq("BENIGN")]
            attacks = chunk[~chunk[LABEL_COL].eq("BENIGN")]

            if not benign.empty:
                file_frames.append(benign.sample(min(len(benign), 1500), random_state=42))
            if not attacks.empty:
                file_frames.append(attacks.sample(min(len(attacks), 1500), random_state=42))

            if sum(len(frame) for frame in file_frames) >= max_rows_per_file:
                break

        if file_frames:
            frames.append(pd.concat(file_frames, ignore_index=True).head(max_rows_per_file))

    if not frames:
        raise ValueError("The dataset did not contain usable rows.")

    data = pd.concat(frames, ignore_index=True)
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    return data


def _prepare_xy(data):
    data = _clean_columns(data)
    X = data[FEATURES].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    y = data[LABEL_COL].astype(str).str.strip().ne("BENIGN").astype(int)
    return X, y


def train_and_save_model(force=False):
    if MODEL_PATH.exists() and not force:
        artifact = joblib.load(MODEL_PATH)
        if isinstance(artifact, dict) and artifact.get("version") == ARTIFACT_VERSION:
            return artifact

    data = _read_training_rows()
    X, y = _prepare_xy(data)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=120,
                    max_depth=16,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Normal", "Attack"],
        output_dict=True,
        zero_division=0,
    )

    artifact = {
        "version": ARTIFACT_VERSION,
        "model": pipeline,
        "features": FEATURES,
        "feature_medians": X.median(numeric_only=True).fillna(0).to_dict(),
        "confusion_matrix": cm.tolist(),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "report": report,
    }
    joblib.dump(artifact, MODEL_PATH)
    return artifact


def load_model():
    return train_and_save_model(force=False)


def make_input_frame(payload, artifact):
    values = {}
    medians = artifact["feature_medians"]
    for feature in artifact["features"]:
        raw_value = payload.get(feature, medians.get(feature, 0))
        values[feature] = pd.to_numeric(raw_value, errors="coerce")

    frame = pd.DataFrame([values], columns=artifact["features"])
    frame = frame.replace([np.inf, -np.inf], np.nan)
    return frame.fillna(pd.Series(medians))


def explain_prediction(frame, artifact, top_n=6):
    pipeline = artifact["model"]
    imputer = pipeline.named_steps["imputer"]
    classifier = pipeline.named_steps["classifier"]
    features = artifact["features"]

    transformed = imputer.transform(frame)

    if shap is not None:
        try:
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(transformed)
            if isinstance(shap_values, list):
                values = shap_values[1][0]
            elif getattr(shap_values, "ndim", 0) == 3:
                values = shap_values[0, :, 1]
            else:
                values = shap_values[0]
        except Exception:
            values = classifier.feature_importances_ * (transformed[0] - imputer.statistics_)
    else:
        values = classifier.feature_importances_ * (transformed[0] - imputer.statistics_)

    rows = []
    for feature, value, contribution in zip(features, transformed[0], values):
        rows.append(
            {
                "feature": feature,
                "value": float(value),
                "impact": float(contribution),
                "direction": "raises attack risk" if contribution >= 0 else "lowers attack risk",
            }
        )

    rows.sort(key=lambda item: abs(item["impact"]), reverse=True)
    return rows[:top_n]
