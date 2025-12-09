import joblib
import pandas as pd
from flask import Flask, request, render_template

MODEL_PATH = 'quality_prediction_model1.joblib'
model_columns = ['Fertilizer', 'temp', 'N', 'P', 'K']
app = Flask(__name__)

loaded_model = joblib.load("quality_prediction_model1.joblib")

GRADE_MAP = {
    0: 'A (High Quality)',
    1: 'B (Medium Quality)',
    2: 'C (Low Quality)',
    3: 'D (Poor Quality)'
}

    
@app.route('/', methods=['GET', 'POST'])
def predict_quality():
    prediction_message = None
    
    if request.method == 'POST':
        data = {col: float(request.form.get(col)) for col in model_columns}
        df = pd.DataFrame([data], columns=model_columns)
        
        predicted_NUMERICAL = loaded_model.predict(df)[0]

        predicted_grade = GRADE_MAP.get(predicted_NUMERICAL, "Unknown Grade")
        prediction_message = f"The predicted crop quality grade is: {predicted_grade}"

    return render_template('index.html', prediction_message=prediction_message, columns=model_columns)

if __name__ == '__main__':
    app.run(debug=True)