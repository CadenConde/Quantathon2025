import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib  # For saving the scaler

# Load the S&P 500 data
base_dir = "../data"  # Change this to the correct directory
data_path = f"{base_dir}/DailyStockBondFloats.csv"
df = pd.read_csv(data_path, parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# Load the Weekly Change Probability dataset (for comparison later)
weekly_change_prob_path = f"{base_dir}/WeeklyChangeProbability.csv"
weekly_change_df = pd.read_csv(weekly_change_prob_path, parse_dates=["Date"])
weekly_change_df = weekly_change_df.sort_values("Date").reset_index(drop=True)

# Define constants
input_length = 252  # 1 year of trading days

# -----------------------
# 1. Load the trained models and scaler
# -----------------------
model_increase = tf.keras.models.load_model("model_increase.keras")
model_decrease = tf.keras.models.load_model("model_decrease.keras")

# Load the scaler used during training
scaler = joblib.load('scaler.pkl')

# Prepare to store predictions
predicted_inc = []
predicted_dec = []
valid_dates = []

# -----------------------
# 2. Iterate over the provided dates in WeeklyChangeProbability.csv
# -----------------------
for idx, row in weekly_change_df.iterrows():
    target_date = row["Date"]
    
    # Find the closest matching date in the S&P 500 data
    closest_idx = df["Date"].sub(target_date).abs().idxmin()
    
    # Check if there is enough data (1 year of trading days ~252 days)
    if closest_idx + input_length > len(df):
        # Skip dates without enough data (too close to the start or end)
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
    
    # Append predictions to lists
    predicted_inc.append(pred_increase_prob)
    predicted_dec.append(pred_decrease_prob)
    
    # Save the date for the valid prediction
    valid_dates.append(target_date)

# Add predictions to the DataFrame
weekly_change_df["PredictedPrInc"] = np.nan
weekly_change_df["PredictedPrDec"] = np.nan

# Only update the rows that had valid predictions
for date, inc, dec in zip(valid_dates, predicted_inc, predicted_dec):
    weekly_change_df.loc[weekly_change_df["Date"] == date, "PredictedPrInc"] = inc
    weekly_change_df.loc[weekly_change_df["Date"] == date, "PredictedPrDec"] = dec

# -----------------------
# 3. Save the predictions to a new CSV file
# -----------------------
output_path = f"{base_dir}/PredictedWeeklyChangeProbability.csv"
weekly_change_df.to_csv(output_path, index=False)

print(f"Predictions saved to: {output_path}")
