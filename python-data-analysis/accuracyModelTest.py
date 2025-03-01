import tensorflow as tf
import pandas as pd
import numpy as np
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
data_dir = os.path.join(base_dir, "data") 

daily_stock_bond_path = os.path.join(data_dir, "DailyStockBond.csv")
weekly_change_prob_path = os.path.join(data_dir, "WeeklyChangeProbability.csv")

daily_stock_bond_df = pd.read_csv(daily_stock_bond_path)
weekly_change_prob_df = pd.read_csv(weekly_change_prob_path)

daily_stock_bond_df["Date"] = pd.to_datetime(daily_stock_bond_df["Date"])
weekly_change_prob_df["Date"] = pd.to_datetime(weekly_change_prob_df["Date"])

daily_stock_bond_df.sort_values("Date", inplace=True)
weekly_change_prob_df.sort_values("Date", inplace=True)

merged_df = pd.merge(weekly_change_prob_df, daily_stock_bond_df, on="Date", how="inner")


merged_df["S&P500"] = merged_df["S&P500"].replace(",", "", regex=True).astype(float)

merged_df["S&P500 Change"] = merged_df["S&P500"].diff()

merged_df["Actual Movement"] = (merged_df["S&P500 Change"] > 0).astype(int)

y_true = merged_df["Actual Movement"].values 
y_pred_prob = merged_df["PrInc"].values  

y_pred = (y_pred_prob > 0.5).astype(int)

accuracy = tf.keras.metrics.Accuracy()
accuracy.update_state(y_true, y_pred)

accuracy_result = accuracy.result().numpy()

print(f"Prediction Accuracy: {accuracy_result:.2%}")
