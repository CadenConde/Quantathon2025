import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ------------------------------
# 1. Load the Data
# ------------------------------
data_path = "../data/DailyStockBondWithInterpolatedProbabilities.csv"
df = pd.read_csv(data_path, parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# ------------------------------
# 2. Compute Historical Volatility from PrDec and PrInc
# ------------------------------
# Heuristic: if a >20% move occurs, assume variance contribution of (0.20)^2.
# Annualized volatility (sigma) = sqrt(PrDec*(0.20)^2 + PrInc*(0.20)^2) * sqrt(252)
df["HistoricalVol"] = np.sqrt( df["PrDec"]*(0.20**2) + df["PrInc"]*(0.20**2) ) * np.sqrt(252)

# ------------------------------
# 3. Prepare Training Data (2007-2018)
# ------------------------------
# We use the period 2007-01-08 to 2018-12-31 for training.
train_start = pd.Timestamp("2007-01-08")
train_end   = pd.Timestamp("2018-12-31")
train_df = df[(df["Date"] >= train_start) & (df["Date"] <= train_end)].copy()

# For a prediction target, we’ll predict the "next day's" volatility.
# (This is just one way to use past info to predict future risk.)
train_df["FutureVol"] = train_df["HistoricalVol"].shift(-1)
train_df = train_df.dropna()

# Our features: PrDec, PrInc, and HistoricalVol.
features = ["PrDec", "PrInc", "HistoricalVol"]
X = train_df[features].values
y = train_df["FutureVol"].values

# Standardize the features.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split training data (we could use a train/test split, but for simplicity we train on all training period).
model = LinearRegression()
model.fit(X_scaled, y)

# ------------------------------
# 4. Predict Volatility for All Dates
# ------------------------------
# For each day in the full dataset, predict the "risk" (volatility) using our model.
X_all = df[features].values
X_all_scaled = scaler.transform(X_all)
df["PredictedRisk"] = model.predict(X_all_scaled)

# ------------------------------
# 5. Compute Optimal Allocation Using MPT (with bonds as risk-free)
# ------------------------------
expected_stock_return = 0.10   # 10% annual return for stocks
risk_free_rate = 0.01          # 1% annual for bonds

# Simplified formula: w_stocks = (E(R_stock)-R_f) / (PredictedRisk^2 * E(R_stock))
# (Note: This is a heuristic; in standard MPT you would also need risk aversion and more details.)
df["StockAllocation"] = (expected_stock_return - risk_free_rate) / (df["PredictedRisk"]**2 * expected_stock_return)

# Clip allocation between 0 and 1
df["StockAllocation"] = df["StockAllocation"].clip(0, 1)
df["BondAllocation"] = 1 - df["StockAllocation"]

# ------------------------------
# 6. Save Output CSV with New Columns
# ------------------------------
output_path = "../data/OptimalAllocations.csv"
# We'll include Date, PredictedRisk, StockAllocation, BondAllocation
output_cols = ["Date", "PredictedRisk", "StockAllocation", "BondAllocation"]
df[output_cols].to_csv(output_path, index=False)

print(f"Optimal allocations saved to {output_path}")