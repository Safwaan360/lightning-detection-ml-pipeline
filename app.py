"""Flask API for serving trained lightning-classification models."""
import os
from flask import Flask, jsonify, request

from ml_pipeline import load_models, run_inference

app = Flask(__name__)
THRESHOLD = float(os.getenv("LIGHTNING_THRESHOLD", "0.20"))
DEFAULT_MODEL = os.getenv("LIGHTNING_MODEL", "hybrid")

# Trained artefacts are intentionally not committed to Git.
load_models(os.getenv("MODEL_DIR", "models"))


@app.post("/predict")
def predict():
    body = request.get_json(silent=True) or {}
    missing = [key for key in ("id", "starttime", "data") if key not in body]
    if missing:
        return jsonify({"error": f"missing field(s): {', '.join(missing)}"}), 400

    model = body.get("model", DEFAULT_MODEL)
    if model not in {"xgb", "cnn", "hybrid"}:
        return jsonify({"error": f"invalid model: {model}"}), 400

    data = body["data"]
    if not isinstance(data, list) or not data:
        return jsonify({"error": "data must be a non-empty list of ADC samples"}), 400

    try:
        scored = run_inference([data], model_choice=model).iloc[0]
        probability = float(scored["prob_lightning"])
        result = {
            "id": body["id"],
            "starttime": str(body["starttime"]),
            "confidence": round(probability, 4),
            "label": "lightning" if probability >= THRESHOLD else "noise",
            "model": model,
            "quality": {
                "is_saturated": bool(scored["is_saturated"]),
                "is_flat": bool(scored["is_flat"]),
            },
        }
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.get("/health")
def health():
    return jsonify({"status": "ok", "models_loaded": True})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
