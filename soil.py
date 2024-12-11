from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved model, scaler, and label encoder
model = joblib.load('optimized_crop_recommendation_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input data from the form
        data = request.json
        features = [
            float(data['N']),
            float(data['P']),
            float(data['K']),
            float(data['temperature']),
            float(data['humidity']),
            float(data['ph']),
            float(data['rainfall']),
        ]

        # Preprocess and predict
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)
        crop_name = label_encoder.inverse_transform(prediction)

        return jsonify({'recommended_crop': crop_name[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
