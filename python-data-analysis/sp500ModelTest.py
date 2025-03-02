import joblib
import numpy as np
import tensorflow as tf
import pandas as pd

# Load the trained model and scaler
model = tf.keras.models.load_model("sp500_allocation_model.keras")
scaler = joblib.load("scaler.pkl")

# Load dataset
file_path = "data/DailyStockBondWithInterpolatedProbabilities.csv"
df = pd.read_csv(file_path)

# Convert Date column to datetime format
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

# Filter test data (2019-2022)
test_df = df[(df["Date"] >= "2019-01-01") & (df["Date"] <= "2022-12-31")].copy()

# Prepare test inputs
X_test = test_df[["PrInc", "PrDec", "Bond Rate"]].values
X_test_scaled = scaler.transform(X_test)  # Normalize

# Predict allocation decisions
test_df["S&P500 Allocation"] = model.predict(X_test_scaled).flatten()

# Define buy/sell signals based on allocation threshold
test_df["Signal"] = np.where(test_df["S&P500 Allocation"] > 0.5, "BUY", "SELL")

# Save predictions
test_df[["Date", "S&P500 Allocation", "Signal"]].to_csv("predictions_2019_2022.csv", index=False)

# Display a sample of predictions
print(test_df[["Date", "S&P500 Allocation", "Signal"]].head(10))