# predict_aqi.py

from tensorflow.keras.models import load_model
import joblib
import numpy as np

# Load the model and scaler
model = load_model('air_quality_model.h5')
scaler = joblib.load('scaler.joblib')

# Sample new data for prediction (replace with actual new data)
new_data = np.array([[12, 34, 25, 60, 0.4]])  # Example values for ['pm2.5', 'pm10', 'temperature', 'relativehumidity', 'um003']

# Preprocess the new data
new_data_scaled = scaler.transform(new_data)
new_data_scaled = new_data_scaled.reshape(new_data_scaled.shape[0], 1, new_data_scaled.shape[1])

# Make a prediction
prediction = model.predict(new_data_scaled)
predicted_label = int(prediction[0] > 0.5)  # Convert sigmoid output to binary label (0: Good, 1: Bad)

# Display the prediction
if predicted_label == 0:
    print("Predicted AQI Level: Good")
else:
    print("Predicted AQI Level: Bad")
