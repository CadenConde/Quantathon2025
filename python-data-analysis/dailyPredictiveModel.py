import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

# Load dataset
file_path = "data/DailyStockBondWithInterpolatedProbabilities.csv"
df = pd.read_csv(file_path)

# Convert Date column to datetime
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

# Split data (Train: 2007-2018, Test: 2019+)
train_df = df[(df["Date"] >= "2007-01-01") & (df["Date"] <= "2018-12-31")]

# Compute returns
train_df["S&P500 Return"] = train_df["S&P500"].pct_change()
train_df["Bond Daily Return"] = train_df["Bond Rate"] / 25200
train_df = train_df.dropna()

# Define input features and target
X_train = train_df[["PrInc", "PrDec", "Bond Rate"]].values  # Features
y_train = train_df["S&P500 Return"].values  # Target (S&P 500 return)

# Scale input features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Build a neural network model
model = Sequential([
    Input(shape=(3,)),  # 3 input features: PrInc, PrDec, Bond Rate
    Dense(16, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')  # Output: allocation weight (0 to 1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
model.fit(X_train_scaled, y_train, epochs=50, batch_size=16, verbose=1)

# Save the scaler and model
joblib.dump(scaler, "scaler.pkl")
model.save("sp500_allocation_model.keras")

print("Done")