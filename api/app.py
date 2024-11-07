# app.py
from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load model
with open('model/model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

app = Flask(__name__)

@app.route('/predict-risk', methods=['POST'])
def predict_risk():
    data = request.json
    features = np.array([[data['age'], data['blood_pressure'], data['cholesterol'], data['smoking'], data['exercise']]])
    risk_score = model.predict_proba(features)[0, 1]
    return jsonify({'risk_score': risk_score, 'recommendation': "Consider regular check-ups."})

if __name__ == '__main__':
    app.run(debug=True)
