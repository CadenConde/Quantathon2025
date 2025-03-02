import pandas as pd

# Load your dataset with S&P500 values and probabilities
data_path = "../data/DailyStockBondWithInterpolatedProbabilities.csv"
df = pd.read_csv(data_path, parse_dates=["Date"])

# Sort data just in case
df = df.sort_values("Date").reset_index(drop=True)

# Compute 12-month future return (252 trading days ahead)
df["FutureS&P500"] = df["S&P500"].shift(-252)
df["FutureReturn"] = (df["FutureS&P500"] - df["S&P500"]) / df["S&P500"]

# Define actual events
df["ActualInc"] = (df["FutureReturn"] >= 0.20).astype(int)  # 1 if S&P500 increased 20%+, else 0
df["ActualDec"] = (df["FutureReturn"] <= -0.20).astype(int)  # 1 if S&P500 decreased 20%+, else 0

# Drop rows where we don't have future data
df = df.dropna()

from sklearn.metrics import brier_score_loss

brier_inc = brier_score_loss(df["ActualInc"], df["PrInc"])
brier_dec = brier_score_loss(df["ActualDec"], df["PrDec"])

print(f"Brier Score for PrInc: {brier_inc:.4f}")  # Lower is better
print(f"Brier Score for PrDec: {brier_dec:.4f}")  # Lower is better