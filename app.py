from pathlib import Path

from flask import Flask, jsonify, redirect, request, send_from_directory
from flask_cors import CORS

from model import explain_prediction, load_model, make_input_frame


app = Flask(__name__)
CORS(app)

artifact = load_model()
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@app.route("/")
def home():
    return redirect("/ui")


@app.route("/ui")
def ui():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/api/status")
def status():
    return jsonify(
        {
            "status": "Cloud IDS backend running",
            "model_version": artifact["version"],
            "features": artifact["features"],
        }
    )


@app.route("/features", methods=["GET"])
def features():
    return jsonify(
        {
            "features": artifact["features"],
            "defaults": artifact["feature_medians"],
            "accuracy": artifact["accuracy"],
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True) or {}
        frame = make_input_frame(payload, artifact)
        model = artifact["model"]

        prediction = int(model.predict(frame)[0])
        probabilities = model.predict_proba(frame)[0]
        attack_probability = float(probabilities[1])

        return jsonify(
            {
                "prediction": "Attack" if prediction == 1 else "Normal",
                "attack_probability": attack_probability,
                "normal_probability": float(probabilities[0]),
                "explanation": explain_prediction(frame, artifact),
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/confusion", methods=["GET"])
def get_confusion():
    return jsonify(
        {
            "matrix": artifact["confusion_matrix"],
            "labels": ["Normal", "Attack"],
            "accuracy": artifact["accuracy"],
            "report": artifact["report"],
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
