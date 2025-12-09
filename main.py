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
# CHATBOT (Gemini API)
# -------------------------------

# ------------------------------------
# FLASK INITIALIZATION
# ------------------------------------
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ------------------------------------
# MODEL 1: QUALITY PREDICTION
# ------------------------------------
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


@app.route("/predict", methods=["POST"])
def predict_quality():
    data = request.get_json() or {}
    try:
        input_data = {col: float(data[col]) for col in QUALITY_COLUMNS}
    except Exception as e:
        return jsonify({"error": f"Invalid or missing fields: {str(e)}"}), 400

    df = pd.DataFrame([input_data], columns=QUALITY_COLUMNS)
    predicted_class = quality_model.predict(df)[0]
    grade = QUALITY_GRADE_MAP.get(predicted_class, "Unknown Grade")

    return jsonify({"grade": grade, "numeric": int(predicted_class)})


# ------------------------------------
# MODEL 2: PRICE PREDICTION
# ------------------------------------
PRICE_MODEL_PATH = "price_prediction_model (2).joblib"
PRICE_COLUMNS_PATH = "model_columns (1).json"

with open(PRICE_COLUMNS_PATH) as f:
    PRICE_COLUMNS = json.load(f)

price_model = joblib.load(PRICE_MODEL_PATH)


@app.route("/predict-price", methods=["POST"])
def predict_price():
    data = request.get_json() or {}

    required = ["Commodity", "Variety", "Grade", "day", "month", "year"]
    missing = [k for k in required if k not in data]

    if missing:
        return jsonify({"error": f"Missing keys: {missing}"}), 400

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

    df_encoded = pd.get_dummies(df)
    df_encoded = df_encoded.reindex(columns=PRICE_COLUMNS, fill_value=0)

    price = float(price_model.predict(df_encoded)[0])

    return jsonify({"price": price})


# ------------------------------------
# MODEL 3: DISEASE PREDICTION
# ------------------------------------
DISEASE_MODEL_PATH = "model_files/krishi_neta_disease_predictor1.keras"

DISEASE_CLASS_NAMES = [
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
]

DISEASE_IMAGE_SIZE = (256, 256)
DISEASE_ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

disease_model = tf.keras.models.load_model(DISEASE_MODEL_PATH, compile=False)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in DISEASE_ALLOWED_EXTENSIONS


@app.route("/predict-disease", methods=["POST"])
def predict_disease():
    if "file" not in request.files:
        return jsonify({"error": "No file found"}), 400

    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid or missing file"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
    file.save(filepath)

    try:
        img = Image.open(filepath).convert("RGB")
        img = img.resize(DISEASE_IMAGE_SIZE)
        img_array = tf.keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        preds = disease_model.predict(img_array)[0]
        idx = int(np.argmax(preds))
        confidence = float(np.max(preds) * 100)
        result = DISEASE_CLASS_NAMES[idx]

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(filepath)

    return jsonify({"prediction": result, "confidence": confidence})


# ------------------------------------
# CHATBOT ENDPOINT (GEMINI API)
# ------------------------------------


# ------------------------------------
# ROOT ENDPOINT
# ------------------------------------
@app.route("/", methods=["GET"])
def root():
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

# ------------------------------------
# MAIN ENTRY
# ------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001, debug=True)
