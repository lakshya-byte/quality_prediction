import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS   # ✅ add this

MODEL_PATH = 'quality_prediction_model1.joblib'
model_columns = ['Fertilizer', 'temp', 'N', 'P', 'K']

app = Flask(__name__)
CORS(app)  # ✅ allow all origins for MVP (you can restrict later)

loaded_model = joblib.load(MODEL_PATH)

GRADE_MAP = {
    0: 'A (High Quality)',
    1: 'B (Medium Quality)',
    2: 'C (Low Quality)',
    3: 'D (Poor Quality)'
}

@app.route('/predict', methods=['POST'])
def predict_quality():
    data = request.get_json()

    input_data = {col: float(data[col]) for col in model_columns}
    df = pd.DataFrame([input_data], columns=model_columns)

    predicted_class = loaded_model.predict(df)[0]
    grade = GRADE_MAP.get(predicted_class, "Unknown Grade")

    return jsonify({
        "grade": grade,
        "numeric": int(predicted_class)
    })


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)
