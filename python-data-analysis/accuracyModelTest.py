# This file tests the accuracy of the provided Probabilities

import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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

# Convert S&P500 column to numeric (removing commas if present)
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

# Extract input features (PrInc, PrDec) and target variable
X = merged_df[["PrInc", "PrDec"]].values
y = merged_df["Market Movement"].values

# Convert -1, 0, 1 labels to a binary classification (increase vs. decrease/stable)
y_binary = (y == 1).astype(int)  # 1 if strong increase, 0 otherwise

# Train logistic regression model
model = LogisticRegression()
model.fit(X, y_binary)

# Make predictions
y_pred = model.predict(X)

# Compute accuracy
accuracy = accuracy_score(y_binary, y_pred)

print(f"Model Accuracy: {accuracy:.2%}")