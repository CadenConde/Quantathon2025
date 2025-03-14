import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# 1. Load Data from OptimalAllocations_Optimized and DailyStockBondFloats
# ------------------------------
optimal_path = "../data/OptimalAllocations_Optimized.csv"  # File containing the optimized stock allocation
daily_path = "../data/DailyStockBondFloats.csv"  # Contains 'S&P500' data

# Load the data
df = pd.read_csv(optimal_path, parse_dates=["Date"])
daily_df = pd.read_csv(daily_path, parse_dates=["Date"])

# Merge on Date (only keep dates present in both datasets)
merged_df = pd.merge(df, daily_df[["Date", "S&P500"]], on="Date", how="inner")

# ------------------------------
# 2. Filter Simulation Period: 2019-01-01 to 2022-12-31
# ------------------------------
start_date = pd.Timestamp("2019-01-01")
end_date   = pd.Timestamp("2022-12-31")
sim_df = merged_df[(merged_df["Date"] >= start_date) & (merged_df["Date"] <= end_date)].copy()
sim_df = sim_df.sort_values("Date").reset_index(drop=True)

# ------------------------------
# 3. Calculate Daily Returns for Stocks and Define Bond Return
# ------------------------------
# Daily S&P500 return as percentage change (first day return set to 0)
sim_df["StockReturn"] = sim_df["S&P500"].pct_change().fillna(0)

# Bond daily return assuming a 1% annual yield (compounded over 252 trading days)
bond_daily_return = (1.01)**(1/252) - 1

# ------------------------------
# 4. Simulate Dynamic Rebalanced Portfolio (Daily Rebalancing) using Optimal Allocation
# ------------------------------
initial_value = 1000.0
portfolio_value_optimal = initial_value
portfolio_value_buyhold = initial_value
dynamic_values_optimal = []
dynamic_values_buyhold = []

# Loop over simulation days; each day rebalance according to optimal allocations.
for idx, row in sim_df.iterrows():
    stock_alloc_optimal = row["StockAllocation_Optimal"]
    
    # Daily return for the optimal strategy
    daily_return_optimal = stock_alloc_optimal * row["StockReturn"] + (1 - stock_alloc_optimal) * bond_daily_return
    
    # Update portfolio value for optimal strategy
    portfolio_value_optimal *= (1 + daily_return_optimal)
    dynamic_values_optimal.append(portfolio_value_optimal)
    
    # For Buy-and-Hold: Invest $1000 on the first day and hold until end.
    if idx == 0:
        initial_sp500 = row["S&P500"]
    
    daily_return_buyhold = 1000 * (row["S&P500"] / initial_sp500)
    portfolio_value_buyhold = daily_return_buyhold
    dynamic_values_buyhold.append(portfolio_value_buyhold)

# Store the dynamic portfolio values for each strategy in the dataframe
sim_df["DynamicPortfolioValue_Optimal"] = dynamic_values_optimal
sim_df["DynamicPortfolioValue_BuyHold"] = dynamic_values_buyhold

# ------------------------------
# 5. Calculate Sharpe Ratio for Each Strategy
# ------------------------------
# Risk-free rate (annualized, assume 1% per year, converted to daily)
risk_free_rate = 0.01 / 252  # 1% annual rate converted to daily

# Calculate daily returns for each strategy
sim_df["Return_Optimal"] = sim_df["DynamicPortfolioValue_Optimal"].pct_change().fillna(0)
sim_df["Return_BuyHold"] = sim_df["DynamicPortfolioValue_BuyHold"].pct_change().fillna(0)

# Function to calculate Sharpe Ratio
def calculate_sharpe_ratio(returns, risk_free_rate):
    # Excess returns = portfolio return - risk-free rate
    excess_returns = returns - risk_free_rate
    # Annualize the Sharpe ratio
    annualized_sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
    return annualized_sharpe

# Compute Sharpe Ratios for each strategy
sharpe_optimal = calculate_sharpe_ratio(sim_df["Return_Optimal"], risk_free_rate)
sharpe_buyhold = calculate_sharpe_ratio(sim_df["Return_BuyHold"], risk_free_rate)

# ------------------------------
# 6. Plot the Results
# ------------------------------
plt.figure(figsize=(12, 6))

plt.plot(sim_df["Date"], sim_df["DynamicPortfolioValue_BuyHold"], label="Buy-and-Hold (S&P 500)", color="tab:blue")
plt.plot(sim_df["Date"], sim_df["DynamicPortfolioValue_Optimal"], label="Optimal Allocation", color="tab:orange")

# Add labels and title
plt.xlabel("Date")
plt.ylabel("Portfolio Value ($)")
plt.title("Optimal Allocation vs. Buy-and-Hold (2019-2022)")
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------
# 7. Print Final Portfolio Values and Sharpe Ratios
# ------------------------------
final_optimal = sim_df["DynamicPortfolioValue_Optimal"].iloc[-1]
final_buy_hold = sim_df["DynamicPortfolioValue_BuyHold"].iloc[-1]

print(f"Final Portfolio Value (Optimal Allocation): ${final_optimal:.2f}")
print(f"Final Portfolio Value (Buy-and-Hold S&P500): ${final_buy_hold:.2f}")

# Print Sharpe Ratios
print(f"Sharpe Ratio (Optimal Allocation): {sharpe_optimal:.8f}")
print(f"Sharpe Ratio (Buy-and-Hold S&P500): {sharpe_buyhold:.8f}")
