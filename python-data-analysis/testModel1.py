import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib  # For saving the scaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load the S&P 500 data
base_dir = "../data"  # Change this to the correct directory
data_path = f"{base_dir}/DailyStockBondFloats.csv"
df = pd.read_csv(data_path, parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# Load the Weekly Change Probability dataset (for comparison later)
weekly_change_prob_path = f"{base_dir}/WeeklyChangeProbability.csv"
weekly_change_df = pd.read_csv(weekly_change_prob_path, parse_dates=["Date"])

weekly_change_df = weekly_change_df.sort_values("Date", ascending=False).reset_index(drop=True)

# Define constants
input_length = 252  # 1 year of trading days
forecast_horizon = 252  # 1 year ahead
drop_threshold = 0.8   # For decrease prediction (20% drop)
rise_threshold = 1.2   # For increase prediction (20% rise)

# -----------------------
# 1. Create Sequences and Labels for Both Models
# -----------------------
X, y_increase, y_decrease = [], [], []

for i in range(len(df) - input_length - forecast_horizon + 1):
    seq = df["S&P500"].iloc[i:i + input_length].values
    current_value = df["S&P500"].iloc[i + input_length - 1]
    future_value = df["S&P500"].iloc[i + input_length - 1 + forecast_horizon]
    
    # Label for Model 1 (Increase)
    label_increase = (future_value >= current_value * rise_threshold)
    y_increase.append(label_increase)
    
    # Label for Model 2 (Decrease)
    label_decrease = (future_value <= current_value * drop_threshold)
    y_decrease.append(label_decrease)
    
    X.append(seq)

X = np.array(X)
y_increase = np.array(y_increase)
y_decrease = np.array(y_decrease)

# -----------------------
# 2. Data Preprocessing (Scaling)
# -----------------------
# Normalize the data
scaler = StandardScaler()
X_scaled = []

for seq in X:
    seq_scaled = scaler.fit_transform(seq.reshape(-1, 1)).flatten()
    X_scaled.append(seq_scaled)

X_scaled = np.array(X_scaled)
X_scaled = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))  # Reshape for LSTM input

# Save the scaler for later use
joblib.dump(scaler, 'scaler.pkl')

# -----------------------
# 3. Train/Test Split
# -----------------------
# Split the data into train/test (80/20 split)
split_index = int(0.8 * len(X_scaled))

X_train, y_train_increase, y_train_decrease = X_scaled[:split_index], y_increase[:split_index], y_decrease[:split_index]
X_test, y_test_increase, y_test_decrease = X_scaled[split_index:], y_increase[split_index:], y_decrease[split_index:]

# -----------------------
# 4. Model 1: Predict Probability of 20% Increase
# -----------------------
model_increase = Sequential([
    LSTM(64, return_sequences=True, input_shape=(input_length, 1)),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification
])

model_increase.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_increase.summary()

# Train Model 1
history_increase = model_increase.fit(X_train, y_train_increase, epochs=15, batch_size=32,
                                      validation_data=(X_test, y_test_increase))

# Save the model
model_increase.save("model_increase.keras")

# -----------------------
# 5. Model 2: Predict Probability of 20% Decrease
# -----------------------
model_decrease = Sequential([
    LSTM(64, return_sequences=True, input_shape=(input_length, 1)),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification
])

model_decrease.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_decrease.summary()

# Train Model 2
history_decrease = model_decrease.fit(X_train, y_train_decrease, epochs=15, batch_size=32,
                                      validation_data=(X_test, y_test_decrease))

# Save the model
model_decrease.save("model_decrease.keras")

# -----------------------
# 6. Make Predictions for Dates in WeeklyChangeProbability.csv
# -----------------------
# Iterate over the provided dates in WeeklyChangeProbability.csv
for idx, row in weekly_change_df.iterrows():
    target_date = row["Date"]
    
    # Find the closest matching date in the S&P 500 data
    closest_idx = df["Date"].sub(target_date).abs().idxmin()
    
    # Check if there is enough data (1 year of trading days ~252 days)
    if closest_idx + input_length > len(df):
        print(f"Not enough data starting from {target_date}.")
        continue
    
    # Extract the 1-year sequence (252 days)
    sample_seq = df["S&P500"].iloc[closest_idx: closest_idx + input_length].values
    
    # Scale the data (using the same scaler used for training)
    sample_seq_scaled = scaler.transform(sample_seq.reshape(-1, 1)).flatten()
    
    # Reshape to match the model's input shape: (1, input_length, 1)
    sample_seq_scaled = sample_seq_scaled.reshape((1, input_length, 1))
    
    # Predict probabilities for both models (increase and decrease)
    pred_increase_prob = model_increase.predict(sample_seq_scaled)[0, 0]
    pred_decrease_prob = model_decrease.predict(sample_seq_scaled)[0, 0]
    
    # Output model predictions and actual values from the dataset
    print(f"Target Date: {target_date}")
    print(f"Predicted Probability of 20% Increase: {pred_increase_prob:.4f}, True Probability: {row['PrInc']:.4f}")
    print(f"Predicted Probability of 20% Decrease: {pred_decrease_prob:.4f}, True Probability: {row['PrDec']:.4f}")
    print("-" * 50)

# -----------------------
# 7. Save Models and Scaler
# -----------------------
model_increase.save("model_increase.keras")
model_decrease.save("model_decrease.keras")
joblib.dump(scaler, 'scaler.pkl')

print("Models and scaler have been saved.")
