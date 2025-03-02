import pandas as pd
import numpy as np
from scipy.optimize import minimize

# ------------------------------
# Load the data
# ------------------------------
input_file = "../data/OptimalAllocations.csv"  # Ensure this file contains 'Date' and 'PredictedRisk' columns
stock_data_file = "../data/DailyStockBondFloats.csv"  # This file contains 'Date' and 'S&P500' columns
output_file = "../data/OptimalAllocations_WithOptimizedRiskLevels.csv"

df = pd.read_csv(input_file, parse_dates=["Date"])
df.sort_values("Date", inplace=True)
df.reset_index(drop=True, inplace=True)

# Load the stock data to get S&P 500 prices
stock_data_df = pd.read_csv(stock_data_file, parse_dates=["Date"])

# ------------------------------
# Merge the dataframes on 'Date' column
# ------------------------------
merged_df = pd.merge(df, stock_data_df[["Date", "S&P500"]], on="Date", how="inner")

# ------------------------------
# Calculate Daily Stock Returns
# ------------------------------
merged_df["StockReturn"] = merged_df["S&P500"].pct_change().fillna(0)  # Daily return of the S&P 500

# ------------------------------
# Define expected returns and risk-free rate
# ------------------------------
expected_stock_return = 0.10  # 10% annual return for stocks
risk_free_rate = 0.01         # 1% annual risk-free rate

# ------------------------------
# Function to calculate the annualized Sharpe ratio
# ------------------------------
def annualized_sharpe_ratio(portfolio_returns, risk_free_rate=0.01):
    """
    Compute the annualized Sharpe ratio for a given portfolio return series.
    
    :param portfolio_returns: Daily returns of the portfolio
    :param risk_free_rate: Risk-free rate (annualized)
    :return: Annualized Sharpe ratio
    """
    excess_returns = portfolio_returns - (risk_free_rate / 252)  # Convert risk-free rate to daily
    mean_excess_return = excess_returns.mean()
    volatility = excess_returns.std()
    return mean_excess_return / volatility * np.sqrt(252)  # Annualized Sharpe ratio

# ------------------------------
# Function to compute stock allocation based on predicted risk and risk aversion coefficient A
# ------------------------------
def compute_allocation(predicted_risk, A):
    if predicted_risk == 0:
        return 1.0  # If predicted risk is zero, allocate 100% to stocks
    allocation = (expected_stock_return - risk_free_rate) / (A * (predicted_risk ** 2) * expected_stock_return)
    return np.clip(allocation, 0, 1)

# ------------------------------
# Objective function to optimize Sharpe ratio (maximize Sharpe ratio)
# ------------------------------
def objective_function(A, predicted_risk, stock_returns, bond_daily_return):
    """
    Objective function to maximize Sharpe ratio by adjusting the risk aversion coefficient.
    
    :param A: Risk aversion coefficient
    :param predicted_risk: Predicted risk (volatility) of the stock
    :param stock_returns: Daily stock returns
    :param bond_daily_return: Daily return of the bond (risk-free asset)
    :return: Negative Sharpe ratio (for minimization)
    """
    # Calculate stock allocation using the given risk aversion coefficient
    stock_allocation = compute_allocation(predicted_risk, A)
    bond_allocation = 1 - stock_allocation

    # Portfolio return is the weighted sum of stock and bond returns
    portfolio_returns = stock_allocation * stock_returns + bond_allocation * bond_daily_return
    return -annualized_sharpe_ratio(portfolio_returns)  # Minimize negative Sharpe ratio

# ------------------------------
# 5. Optimize Risk Aversion Coefficient (A) for Each Day in 2019-2022
# ------------------------------
bond_daily_return = (1.01)**(1/252) - 1  # Assume a 1% annual return for bonds

best_A_values = []
stock_allocation_values = []
bond_allocation_values = []
sharpe_ratios = []

for idx, row in merged_df.iterrows():
    predicted_risk = row["PredictedRisk"]
    stock_returns = merged_df["StockReturn"].iloc[max(0, idx-252):idx+1]  # Use past year of stock returns for estimation

    # Optimize risk aversion coefficient A for each day
    result = minimize(objective_function, x0=[1], bounds=[(0.1, 10)], args=(predicted_risk, stock_returns, bond_daily_return))
    
    best_A_values.append(result.x[0])
    
    # Compute optimal stock allocation based on the optimized A
    stock_allocation = compute_allocation(predicted_risk, result.x[0])
    bond_allocation = 1 - stock_allocation
    
    stock_allocation_values.append(stock_allocation)
    bond_allocation_values.append(bond_allocation)
    sharpe_ratios.append(-result.fun)  # Negative Sharpe ratio (since we're minimizing)

# ------------------------------
# 6. Save Results to CSV
# ------------------------------
merged_df["BestRiskAversionCoefficient"] = best_A_values
merged_df["StockAllocationOptimal"] = stock_allocation_values
merged_df["BondAllocationOptimal"] = bond_allocation_values

# Save the results to a new CSV file
merged_df.to_csv(output_file, index=False)
print(f"Optimal allocations with dynamic risk levels and optimized risk aversion coefficients saved to: {output_file}")

# ------------------------------
# Print Optimal Risk Aversion and Sharpe Value to Console
# ------------------------------
optimal_A = best_A_values[-1]  # The last value will be the optimized A for the last date
final_sharpe_ratio = sharpe_ratios[-1]  # The last Sharpe ratio

print(f"Optimal Risk Aversion Coefficient: {optimal_A:.4f}")
print(f"Final Sharpe Ratio: {final_sharpe_ratio:.4f}")
