import tensorflow as tf
import joblib
import pandas as pd
import numpy as np

# Define the custom Sharpe Ratio loss function
def sharpe_loss(y_true, y_pred):
    portfolio_return = y_pred * y_true  # Portfolio return = allocation * S&P 500 return
    return -tf.reduce_mean(portfolio_return) / (tf.keras.backend.std(portfolio_return) + 1e-6)

# Load trained model with custom loss function
try:
    model = tf.keras.models.load_model("sp500_allocation_model.keras", custom_objects={"sharpe_loss": sharpe_loss})
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# Load scaler
try:
    scaler = joblib.load("scaler.pkl")
    print("✅ Scaler loaded successfully.")
except Exception as e:
    print(f"❌ Error loading scaler: {e}")
    exit()

# Load dataset
file_path = "data/DailyStockBondWithInterpolatedProbabilities.csv"
try:
    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    print("✅ Dataset loaded and processed.")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    exit()

# Filter test data (2019-2022)
test_df = df[(df["Date"] >= "2019-01-01") & (df["Date"] <= "2022-12-31")].copy()

# Ensure required columns exist
required_cols = ["PrInc", "PrDec", "Bond Rate"]
if not all(col in test_df.columns for col in required_cols):
    print(f"❌ Error: Missing required columns in dataset: {required_cols}")
    exit()

# Prepare test inputs
X_test = test_df[required_cols].values
X_test_scaled = scaler.transform(X_test)  # Normalize inputs

# Run model predictions
test_df["S&P500 Allocation"] = model.predict(X_test_scaled).flatten()

# Define buy/sell signals based on allocation threshold
test_df["Signal"] = np.where(test_df["S&P500 Allocation"] > 0.5, "BUY", "SELL")

# Save results to CSV
output_path = "predictions_2019_2022.csv"
test_df[["Date", "S&P500 Allocation", "Signal"]].to_csv(output_path, index=False)

# Print first few rows
print("\n📊 Sample Predictions:")
print(test_df[["Date", "S&P500 Allocation", "Signal"]].head(10))

print(f"\n✅ Predictions saved successfully: {output_path}")