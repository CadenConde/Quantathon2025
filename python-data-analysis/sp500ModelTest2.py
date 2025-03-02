import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

# Load dataset
file_path = "data/DailyStockBondWithInterpolatedProbabilities.csv"
df = pd.read_csv(file_path)

# Convert Date column to datetime
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

# Filter data from 2007 through 2018
filtered_df = df[(df["Date"] >= "2007-01-01") & (df["Date"] <= "2018-12-31")].copy()

# Load trained model and scaler
model = tf.keras.models.load_model("sp500_allocation_model.keras")
scaler = joblib.load("scaler.pkl")

# Prepare input features
X_filtered = filtered_df[["PrInc", "PrDec", "Bond Rate"]].values
X_filtered_scaled = scaler.transform(X_filtered)  # Normalize input data

# Run model predictions
filtered_df["S&P500 Allocation"] = model.predict(X_filtered_scaled).flatten()

# Define buy/sell signals
filtered_df["Signal"] = np.where(filtered_df["S&P500 Allocation"] > 0.5, "BUY", "SELL")

# Save results to CSV
output_path = "predictions_2007_2018.csv"
filtered_df[["Date", "S&P500 Allocation", "Signal"]].to_csv(output_path, index=False)

# Print first few rows
print(filtered_df[["Date", "S&P500 Allocation", "Signal"]].head())

print(f"\n✅ Predictions saved successfully: {output_path}")