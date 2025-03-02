import tensorflow as tf
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
required_cols = ["PrInc", "PrDec", "Bond Rate", "S&P500"]
if not all(col in test_df.columns for col in required_cols):
    print(f"❌ Error: Missing required columns in dataset: {required_cols}")
    exit()

# Compute daily returns
test_df["S&P500 Return"] = test_df["S&P500"].pct_change().fillna(0)  # Daily % change for stocks
test_df["Bond Daily Return"] = test_df["Bond Rate"] / 25200  # Convert annual bond rate to daily return

# Prepare test inputs
X_test = test_df[["PrInc", "PrDec", "Bond Rate"]].values
X_test_scaled = scaler.transform(X_test)  # Normalize inputs

# Run model predictions
test_df["Stock Allocation"] = model.predict(X_test_scaled).flatten()

# Ensure allocations are between 0 and 1
test_df["Stock Allocation"] = test_df["Stock Allocation"].clip(0, 1)

# Compute Bond Allocation
test_df["Bond Allocation"] = 1 - test_df["Stock Allocation"]

# Convert allocations to percentages
test_df["Stock Allocation (%)"] = (test_df["Stock Allocation"] * 100).round(2)
test_df["Bond Allocation (%)"] = (test_df["Bond Allocation"] * 100).round(2)

# Define buy/sell signals based on allocation threshold
test_df["Signal"] = np.where(test_df["Stock Allocation"] > 0.5, "BUY", "SELL")

# Simulate portfolio value starting with $10,000
initial_investment = 10000
portfolio_value = initial_investment
portfolio_values = []

for _, row in test_df.iterrows():
    # Calculate daily return based on allocation
    daily_stock_return = row["Stock Allocation"] * row["S&P500 Return"]
    daily_bond_return = row["Bond Allocation"] * row["Bond Daily Return"]
    
    # Compute portfolio growth for the day
    daily_portfolio_return = daily_stock_return + daily_bond_return
    portfolio_value *= (1 + daily_portfolio_return)
    
    # Store the portfolio value
    portfolio_values.append(portfolio_value)

# Add portfolio value to DataFrame
test_df["Portfolio Value ($)"] = portfolio_values

# Save results to CSV
output_path = "predictions_2019_2022.csv"
test_df[["Date", "Stock Allocation (%)", "Bond Allocation (%)", "Signal", "Portfolio Value ($)"]].to_csv(output_path, index=False)

# Plot the portfolio value over time
plt.figure(figsize=(12, 6))
plt.plot(test_df["Date"], test_df["Portfolio Value ($)"], label="Portfolio Value", color="blue", linewidth=2)
plt.xlabel("Date")
plt.ylabel("Portfolio Value ($)")
plt.title("Portfolio Growth Over Time (2019-2022)")
plt.legend()
plt.grid(True)

# Save the plot
plot_path = "portfolio_growth.png"
plt.savefig(plot_path, dpi=300)
plt.show()

print("\n📊 Sample Predictions:")
print(test_df[["Date", "Stock Allocation (%)", "Bond Allocation (%)", "Signal", "Portfolio Value ($)"]].head(10))

print(f"\n✅ Predictions saved successfully: {output_path}")
print(f"📈 Portfolio growth chart saved as: {plot_path}")