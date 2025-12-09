import os
import json

import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf

# -------------------------------
# Flask app setup
# -------------------------------
app = Flask(__name__)
CORS(app)  # allow all origins for MVP (you can restrict later)

# Make sure upload folder exists (for disease model)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# -------------------------------
# MODEL 1: QUALITY PREDICTION
# -------------------------------
QUALITY_MODEL_PATH = "quality_prediction_model1.joblib"
QUALITY_COLUMNS_PATH = "model_columns1.json"

with open(QUALITY_COLUMNS_PATH) as f:
    QUALITY_COLUMNS = json.load(f)

quality_model = joblib.load(QUALITY_MODEL_PATH)

QUALITY_GRADE_MAP = {
    0: "A (High Quality)",
    1: "B (Medium Quality)",
    2: "C (Low Quality)",
    3: "D (Poor Quality)",
}


@app.route("/predict-quality", methods=["POST"])
def predict_quality():
    """
    JSON input:
    {
      "Fertilizer": 10,
      "temp": 30,
      "N": 5,
      "P": 6,
      "K": 7
    }
    """
    data = request.get_json() or {}

    try:
        input_data = {
            col: float(data[col]) for col in QUALITY_COLUMNS
        }
    except KeyError as e:
        return jsonify({"error": f"Missing key: {str(e)}"}), 400
    except ValueError:
        return jsonify({"error": "All inputs must be numeric"}), 400

    df = pd.DataFrame([input_data], columns=QUALITY_COLUMNS)

    predicted_class = quality_model.predict(df)[0]
    grade = QUALITY_GRADE_MAP.get(predicted_class, "Unknown Grade")

    return jsonify(
        {
            "grade": grade,
            "numeric": int(predicted_class),
            "inputs": input_data,
        }
    )


# -------------------------------
# MODEL 2: PRICE PREDICTION
# -------------------------------
PRICE_MODEL_PATH = "price_prediction_model (2).joblib"
PRICE_COLUMNS_PATH = "model_columns (1).json"

with open(PRICE_COLUMNS_PATH) as f:
    PRICE_COLUMNS = json.load(f)

price_model = joblib.load(PRICE_MODEL_PATH)


@app.route("/predict-price", methods=["POST"])
def predict_price():
    """
    JSON input:
    {
      "Commodity": "Rice",
      "Variety": "Basmati",
      "Grade": "A",
      "day": 9,
      "month": 12,
      "year": 2025
    }
    """
    data = request.get_json() or {}

    required_keys = ["Commodity", "Variety", "Grade", "day", "month", "year"]
    missing = [k for k in required_keys if k not in data]
    if missing:
        return jsonify({"error": f"Missing keys: {missing}"}), 400

    try:
        df = pd.DataFrame(
            [
                {
                    "Commodity": data["Commodity"],
                    "Variety": data["Variety"],
                    "Grade": data["Grade"],
                    "day": int(data["day"]),
                    "month": int(data["month"]),
                    "year": int(data["year"]),
                }
            ]
        )
    except ValueError:
        return jsonify({"error": "day/month/year must be integers"}), 400

    # One-hot encode and align with training columns
    df_encoded = pd.get_dummies(df)
    df_encoded = df_encoded.reindex(columns=PRICE_COLUMNS, fill_value=0)

    price = float(price_model.predict(df_encoded)[0])

    return jsonify(
        {
            "price": price,
            "inputs": data,
        }
    )


# -------------------------------
# MODEL 3: DISEASE PREDICTION
# -------------------------------
DISEASE_MODEL_PATH = "model_files/krishi_neta_disease_predictor1.keras"

# Same class names as in disease repo
DISEASE_CLASS_NAMES = [
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
]

DISEASE_IMAGE_SIZE = (256, 256)
DISEASE_ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

disease_model = tf.keras.models.load_model(DISEASE_MODEL_PATH, compile=False)


def allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in DISEASE_ALLOWED_EXTENSIONS
    )


def predict_disease_image(image_path: str):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(DISEASE_IMAGE_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    preds = disease_model.predict(img_array)[0]
    predicted_index = int(np.argmax(preds))
    confidence = float(np.max(preds) * 100.0)

    if predicted_index >= len(DISEASE_CLASS_NAMES):
        raise ValueError("Predicted class index out of range")

    predicted_class = DISEASE_CLASS_NAMES[predicted_index]
    return predicted_class, confidence


@app.route("/predict-disease", methods=["POST"])
def predict_disease():
    """
    Form-data input:
    file: (image file: jpg / png / jpeg)
    """
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        prediction, confidence = predict_disease_image(filepath)
    except Exception as e:
        os.remove(filepath)
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    os.remove(filepath)

    return jsonify(
        {
            "prediction": prediction,
            "confidence": confidence,
        }
    )


# -------------------------------
# Health check / root endpoint
# -------------------------------
@app.route("/", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "endpoints": [
                "/predict",
                "/predict-price",
                "/predict-disease",
            ],
        }
    )


# -------------------------------
# Main entrypoint
# -------------------------------
if __name__ == "__main__":
    # For Docker / EC2
    app.run(host="0.0.0.0", port=8001, debug=True)
