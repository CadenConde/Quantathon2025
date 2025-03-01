import tensorflow as tf
import pandas as pd
import numpy as np
import os

# Define file paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Get project root directory
data_dir = os.path.join(base_dir, "data")  # Path to data folder

daily_stock_bond_path = os.path.join(data_dir, "DailyStockBond.csv")
weekly_change_prob_path = os.path.join(data_dir, "WeeklyChangeProbability.csv")

# Load CSV files
daily_stock_bond_df = pd.read_csv(daily_stock_bond_path)
weekly_change_prob_df = pd.read_csv(weekly_change_prob_path)

# Convert Date columns to datetime
daily_stock_bond_df["Date"] = pd.to_datetime(daily_stock_bond_df["Date"])
weekly_change_prob_df["Date"] = pd.to_datetime(weekly_change_prob_df["Date"])

# Sort data by Date in ascending order
daily_stock_bond_df.sort_values("Date", inplace=True)
weekly_change_prob_df.sort_values("Date", inplace=True)

# Merge datasets on Date
merged_df = pd.merge(weekly_change_prob_df, daily_stock_bond_df, on="Date", how="inner")

# Convert S&P500 column to numeric (removing any commas if present)
merged_df["S&P500"] = merged_df["S&P500"].replace(",", "", regex=True).astype(float)

# Create a column for future S&P500 price (12 months ahead)
merged_df["Future S&P500"] = merged_df["S&P500"].shift(-12)  # Shift by 12 months

# Compute percentage change over 12 months
merged_df["S&P500 12M Change"] = ((merged_df["Future S&P500"] - merged_df["S&P500"]) / merged_df["S&P500"]) * 100

# Define target labels based on 20% thresholds
def classify_movement(change):
    if change >= 20:
        return 1  # Strong Increase
    elif change <= -20:
        return -1  # Strong Decrease
    else:
        return 0  # Stable

merged_df["Market Movement"] = merged_df["S&P500 12M Change"].apply(classify_movement)

# Drop rows with NaN values (caused by shifting)
merged_df.dropna(inplace=True)

# Define input features (PrInc, PrDec) and target variable
X = merged_df[["PrInc", "PrDec"]].values
y = merged_df["Market Movement"].values

# Convert labels to categorical (one-hot encoding)
y_categorical = tf.keras.utils.to_categorical(y + 1, num_classes=3)  # Shift labels to fit 0,1,2

# Define a simple neural network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')  # 3 output classes (-1, 0, 1)
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y_categorical, epochs=50, batch_size=8, verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate(X, y_categorical)
print(f"Model Accuracy: {accuracy:.2%}")