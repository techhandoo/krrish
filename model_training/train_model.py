import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout

# Load dataset
data = pd.read_csv('updated_air_quality_data_with_AQI.csv')

# Label AQI as Good (0) or Bad (1)
data['AQI_label'] = np.where(data['AQI'] < 100, 0, 1)

# Define features and target
features = data[['pm2.5', 'pm10', 'temperature', 'relativehumidity', 'um003']].values
target = data['AQI_label'].values

# Normalize features
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)
features_scaled = features_scaled.reshape(features_scaled.shape[0], 1, features_scaled.shape[1])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

# Define CNN-LSTM model with kernel size 1 for Conv1D
model = Sequential([
    Conv1D(64, 1, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    MaxPooling1D(1),
    LSTM(50, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# Compile and train model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Save the model without batch_shape issues
model.save('air_quality_model.h5', include_optimizer=False)

# Save the scaler
import joblib
joblib.dump(scaler, 'scaler.joblib')
