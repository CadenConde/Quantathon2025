import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import joblib  # Add joblib for saving the scaler



# -----------------------
# 1. Load and Prepare Data
# -----------------------
# Adjust the path to your CSV file as needed.
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, "data")
data_path = os.path.join(data_dir, "DailyStockBondFloats.csv")

# Load the CSV and sort by date.
df = pd.read_csv(data_path, parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# We'll work with the S&P500 column.
prices = df["S&P500"].values
dates = df["Date"].values

# -----------------------
# 2. Create Sequences and Labels
# -----------------------
# Define parameters:
input_length = 252      # ~1 year of trading days
forecast_horizon = 252  # 1 year ahead

X, y = [], []
# For each sample, the input sequence is 252 days of data.
# The "current value" is taken as the last day in the sequence.
# The label is 1 if the price forecast_horizon days after the end is <= 80% of that current value.
for i in range(len(prices) - input_length - forecast_horizon + 1):
    seq = prices[i: i + input_length]
    current_value = prices[i + input_length - 1]
    future_value = prices[i + input_length - 1 + forecast_horizon]
    label = 1 if future_value <= current_value * 0.8 else 0
    X.append(seq)
    y.append(label)

X = np.array(X)
y = np.array(y)
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Positive samples (≥20% drop):", np.sum(y==1), 
      "Negative samples:", np.sum(y==0))

# -----------------------
# 3. Data Normalization
# -----------------------
# We perform global scaling using StandardScaler.
scaler = StandardScaler()
# Fit scaler on all prices (reshaped to 2D).
scaler.fit(prices.reshape(-1, 1))

# Apply the scaler to each input sequence.
X_scaled = []
for seq in X:
    seq_scaled = scaler.transform(seq.reshape(-1, 1)).flatten()
    X_scaled.append(seq_scaled)
X_scaled = np.array(X_scaled)

# Reshape for LSTM: (samples, timesteps, features)
X_scaled = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))

# -----------------------
# 4. Train/Validation/Test Split
# -----------------------
# We do a time-series split.
split_index = int(0.7 * len(X_scaled))  # first 70% for training
X_train, y_train = X_scaled[:split_index], y[:split_index]
X_temp, y_temp = X_scaled[split_index:], y[split_index:]

# Split the remaining 30% equally into validation and test sets.
val_index = int(0.5 * len(X_temp))
X_val, y_val = X_temp[:val_index], y_temp[:val_index]
X_test, y_test = X_temp[val_index:], y_temp[val_index:]

print("Training samples:", X_train.shape[0])
print("Validation samples:", X_val.shape[0])
print("Test samples:", X_test.shape[0])

# -----------------------
# 5. Build and Train a TensorFlow Model
# -----------------------
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(input_length, 1)),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model.
history = model.fit(X_train, y_train, epochs=20, batch_size=32,
                    validation_data=(X_val, y_val))

# -----------------------
# 6. Evaluate the Model
# -----------------------
# Predict probabilities on the test set.
y_pred_prob = model.predict(X_test)
# Convert probabilities to binary predictions.
y_pred = (y_pred_prob >= 0.5).astype(int).flatten()

accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the model
model.save("my_trained_model.h5")

# After scaling the training data, save the scaler
joblib.dump(scaler, 'scaler.pkl')

# -----------------------
# 7. Plot Training History
# -----------------------
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Val Loss")
plt.title("Loss")
plt.legend()
plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label="Train Accuracy")
plt.plot(history.history['val_accuracy'], label="Val Accuracy")
plt.title("Accuracy")
plt.legend()
plt.show()
