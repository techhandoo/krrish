from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import logging

logging.basicConfig(level=logging.DEBUG)
logging.info("Starting the app")


# Load the model and scaler
model = load_model('model_training/air_quality_model.h5')
scaler = np.load('scaler.npy', allow_pickle=True)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from the request
    data = request.get_json()
    pm25 = float(data.get('pm2.5'))
    pm10 = float(data.get('pm10'))
    temperature = float(data.get('temperature'))
    humidity = float(data.get('relativehumidity'))
    um003 = float(data.get('um003'))

    # Scale and reshape the input data
    input_data = scaler.transform([[pm25, pm10, temperature, humidity, um003]])
    input_data = input_data.reshape(1, 1, 5)

    # Make a prediction
    prediction = model.predict(input_data)
    prediction_label = "Good" if prediction < 0.5 else "Bad"
    
    return jsonify({'AQI_prediction': prediction_label})

if __name__ == '__main__':
    app.run(debug=True)

