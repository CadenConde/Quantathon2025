import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the necessary data
base_dir = "../data"  # Change this to the correct directory
predicted_path = f"{base_dir}/PredictedWeeklyChangeProbability.csv"
stock_data_path = f"{base_dir}/DailyStockBondFloats.csv"

# Load the predicted probabilities
predicted_df = pd.read_csv(predicted_path, parse_dates=["Date"])

# Load the actual data (S&P 500 values)
stock_df = pd.read_csv(stock_data_path, parse_dates=["Date"])

# Merge the two datasets on Date to align predictions with actual values
merged_df = pd.merge(predicted_df, stock_df, on="Date", how="inner")

# ------------------------
# Check Predictions vs Actuals
# ------------------------

# Calculate the correctness of predictions for both increase and decrease
merged_df['CorrectPrInc'] = (merged_df['PredictedPrInc'] >= 0.5) == merged_df['PrInc']
merged_df['CorrectPrDec'] = (merged_df['PredictedPrDec'] >= 0.5) == merged_df['PrDec']

# ------------------------
# Plotting: Predictions vs Actuals
# ------------------------

# Plot the predicted probabilities vs actual values
plt.figure(figsize=(14, 6))

# Plot Predicted vs Actual for Increase Probability
plt.subplot(1, 2, 1)
plt.scatter(merged_df['PredictedPrInc'], merged_df['PrInc'], color='blue', alpha=0.6, label='Increase Prediction')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Perfect Prediction')
plt.xlabel('Predicted Probability of 20% Increase')
plt.ylabel('Actual Probability of 20% Increase')
plt.title('Predicted vs Actual Probability of 20% Increase')
plt.legend()

# Plot Predicted vs Actual for Decrease Probability
plt.subplot(1, 2, 2)
plt.scatter(merged_df['PredictedPrDec'], merged_df['PrDec'], color='green', alpha=0.6, label='Decrease Prediction')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Perfect Prediction')
plt.xlabel('Predicted Probability of 20% Decrease')
plt.ylabel('Actual Probability of 20% Decrease')
plt.title('Predicted vs Actual Probability of 20% Decrease')
plt.legend()

# Display the plots
plt.tight_layout()
plt.show()

# ------------------------
# Effectiveness: Accuracy
# ------------------------

# Calculate the accuracy of the predictions
accuracy_inc = np.mean(merged_df['CorrectPrInc'])
accuracy_dec = np.mean(merged_df['CorrectPrDec'])

print(f"Accuracy for predicting 20% increase: {accuracy_inc:.4f}")
print(f"Accuracy for predicting 20% decrease: {accuracy_dec:.4f}")

