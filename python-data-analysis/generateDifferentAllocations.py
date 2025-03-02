import pandas as pd
import numpy as np

# Load the data
input_file = "../data/OptimalAllocations.csv"  # Ensure this file contains 'Date' and 'PredictedRisk' columns
output_file = "../data/OptimalAllocations_Optimized.csv"

df = pd.read_csv(input_file, parse_dates=["Date"])
df.sort_values("Date", inplace=True)
df.reset_index(drop=True, inplace=True)

# Define expected returns and risk-free rate
expected_stock_return = 0.10  # 10% annual return for stocks
risk_free_rate = 0.01         # 1% annual risk-free rate

# Define risk aversion parameters for each level
risk_aversion = {
    # "Low": .5,     # Less risk-averse → higher stock allocation
    # "Medium": 1,
    "Optimal": .685,
    # "High": 2     # More risk-averse → lower stock allocation
}

# Function to compute stock allocation based on predicted risk and risk aversion coefficient A
def compute_allocation(predicted_risk, A):
    if predicted_risk == 0:
        return 1.0  # If predicted risk is zero, allocate 100% to stocks
    allocation = (expected_stock_return - risk_free_rate) / (A * (predicted_risk ** 2) * expected_stock_return)
    return np.clip(allocation, 0, 1)

# Compute allocations for each risk level
for level, A in risk_aversion.items():
    df[f"StockAllocation_{level}"] = df["PredictedRisk"].apply(lambda x: compute_allocation(x, A))
    df[f"BondAllocation_{level}"] = 1 - df[f"StockAllocation_{level}"]

# Save the new CSV file with the additional columns
df.to_csv(output_file, index=False)
print(f"Optimal allocations with dynamic risk levels saved to: {output_file}")
