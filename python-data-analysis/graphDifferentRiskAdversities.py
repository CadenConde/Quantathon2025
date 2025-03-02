import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# 1. Load Data from OptimalAllocations and DailyStockBondFloats
# ------------------------------
optimal_path = "../data/OptimalAllocations_WithDynamicRiskLevels.csv"  # Updated file with dynamic risk levels
daily_path = "../data/DailyStockBondFloats.csv"

# Load the data from the previous files
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
# 4. Simulate Dynamic Rebalanced Portfolio (Daily Rebalancing)
# ------------------------------
initial_value = 1000.0
portfolio_value_low = initial_value
portfolio_value_medium = initial_value
portfolio_value_high = initial_value
dynamic_values_low = []
dynamic_values_medium = []
dynamic_values_high = []

# Loop over simulation days; each day rebalance according to given allocations.
for idx, row in sim_df.iterrows():
    # For each risk level: Low, Medium, High
    stock_alloc_low = row["StockAllocation_Low"]
    stock_alloc_medium = row["StockAllocation_Medium"]
    stock_alloc_high = row["StockAllocation_High"]

    # Daily return for each strategy
    daily_return_low = stock_alloc_low * row["StockReturn"] + (1 - stock_alloc_low) * bond_daily_return
    daily_return_medium = stock_alloc_medium * row["StockReturn"] + (1 - stock_alloc_medium) * bond_daily_return
    daily_return_high = stock_alloc_high * row["StockReturn"] + (1 - stock_alloc_high) * bond_daily_return

    # Update portfolio value for each strategy
    portfolio_value_low *= (1 + daily_return_low)
    portfolio_value_medium *= (1 + daily_return_medium)
    portfolio_value_high *= (1 + daily_return_high)

    # Append the updated portfolio values
    dynamic_values_low.append(portfolio_value_low)
    dynamic_values_medium.append(portfolio_value_medium)
    dynamic_values_high.append(portfolio_value_high)

# Store the dynamic portfolio values for each strategy in the dataframe
sim_df["DynamicPortfolioValue_Low"] = dynamic_values_low
sim_df["DynamicPortfolioValue_Medium"] = dynamic_values_medium
sim_df["DynamicPortfolioValue_High"] = dynamic_values_high

# ------------------------------
# 5. Compute Buy-and-Hold Strategy for S&P 500
# ------------------------------
# Buy-and-hold: Invest $1000 on the first trading day and hold until end.
initial_sp500 = sim_df["S&P500"].iloc[0]
sim_df["BuyHoldValue"] = 1000 * (sim_df["S&P500"] / initial_sp500)

# ------------------------------
# 6. Calculate Sharpe Ratio for Each Strategy
# ------------------------------
# Risk-free rate (annualized, assume 1% per year, converted to daily)
risk_free_rate = 0.01 / 252  # 1% annual rate converted to daily

# Calculate daily returns for each strategy
sim_df["Return_Low"] = sim_df["DynamicPortfolioValue_Low"].pct_change().fillna(0)
sim_df["Return_Medium"] = sim_df["DynamicPortfolioValue_Medium"].pct_change().fillna(0)
sim_df["Return_High"] = sim_df["DynamicPortfolioValue_High"].pct_change().fillna(0)
sim_df["Return_BuyHold"] = sim_df["BuyHoldValue"].pct_change().fillna(0)

# Function to calculate Sharpe Ratio
def calculate_sharpe_ratio(returns, risk_free_rate):
    excess_returns = returns*3 - risk_free_rate
    return excess_returns.mean() / excess_returns.std()

# Compute Sharpe Ratios for each strategy
sharpe_low = calculate_sharpe_ratio(sim_df["Return_Low"], risk_free_rate)
sharpe_medium = calculate_sharpe_ratio(sim_df["Return_Medium"], risk_free_rate)
sharpe_high = calculate_sharpe_ratio(sim_df["Return_High"], risk_free_rate)
sharpe_buyhold = calculate_sharpe_ratio(sim_df["Return_BuyHold"], risk_free_rate)

# ------------------------------
# 7. Plot the Results
# ------------------------------
plt.figure(figsize=(12, 6))
plt.plot(sim_df["Date"], sim_df["DynamicPortfolioValue_Low"], label="Low Risk Aversion", color="green")
plt.plot(sim_df["Date"], sim_df["DynamicPortfolioValue_Medium"], label="Medium Risk Aversion", color="blue")
plt.plot(sim_df["Date"], sim_df["DynamicPortfolioValue_High"], label="High Risk Aversion", color="red")
plt.plot(sim_df["Date"], sim_df["BuyHoldValue"], label="Buy-and-Hold (S&P 500)", color="black", linestyle="--")

# Add labels and title
plt.xlabel("Date")
plt.ylabel("Portfolio Value ($)")
plt.title("Portfolio Value Over Time: Low, Medium, High Risk vs. Buy-and-Hold (2019-2022)")
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------
# 8. Print Final Portfolio Values and Sharpe Ratios
# ------------------------------
final_low = sim_df["DynamicPortfolioValue_Low"].iloc[-1]
final_medium = sim_df["DynamicPortfolioValue_Medium"].iloc[-1]
final_high = sim_df["DynamicPortfolioValue_High"].iloc[-1]
final_buy_hold = sim_df["BuyHoldValue"].iloc[-1]

print(f"Final Portfolio Value (Low Risk Aversion): ${final_low:.2f}")
print(f"Final Portfolio Value (Medium Risk Aversion): ${final_medium:.2f}")
print(f"Final Portfolio Value (High Risk Aversion): ${final_high:.2f}")
print(f"Final Portfolio Value (Buy-and-Hold S&P500): ${final_buy_hold:.2f}")

# Print Sharpe Ratios
print(f"Sharpe Ratio (Low Risk Aversion): {sharpe_low:.8f}")
print(f"Sharpe Ratio (Medium Risk Aversion): {sharpe_medium:.8f}")
print(f"Sharpe Ratio (High Risk Aversion): {sharpe_high:.8f}")
print(f"Sharpe Ratio (Buy-and-Hold S&P500): {sharpe_buyhold:.8f}")




# Annualized return for each strategy
annualized_return_low = sim_df["Return_Low"].mean() * 252  # Assuming 252 trading days per year
annualized_return_medium = sim_df["Return_Medium"].mean() * 252
annualized_return_high = sim_df["Return_High"].mean() * 252
annualized_return_buyhold = sim_df["Return_BuyHold"].mean() * 252

# Standard deviation (risk) of daily returns, annualized
annualized_vol_low = sim_df["Return_Low"].std() * np.sqrt(252)
annualized_vol_medium = sim_df["Return_Medium"].std() * np.sqrt(252)
annualized_vol_high = sim_df["Return_High"].std() * np.sqrt(252)
annualized_vol_buyhold = sim_df["Return_BuyHold"].std() * np.sqrt(252)

# Risk-free rate (assumed to be 2% annualized)
risk_free_rate_annual = 0.01

# Annualized Sharpe ratios for each strategy
sharpe_low_annual = (annualized_return_low - risk_free_rate_annual) / annualized_vol_low
sharpe_medium_annual = (annualized_return_medium - risk_free_rate_annual) / annualized_vol_medium
sharpe_high_annual = (annualized_return_high - risk_free_rate_annual) / annualized_vol_high
sharpe_buyhold_annual = (annualized_return_buyhold - risk_free_rate_annual) / annualized_vol_buyhold

print(f"Annualized Sharpe Ratio (Low Risk Aversion): {sharpe_low_annual:.8f}")
print(f"Annualized Sharpe Ratio (Medium Risk Aversion): {sharpe_medium_annual:.8f}")
print(f"Annualized Sharpe Ratio (High Risk Aversion): {sharpe_high_annual:.8f}")
print(f"Annualized Sharpe Ratio (Buy-and-Hold S&P500): {sharpe_buyhold_annual:.8f}")
