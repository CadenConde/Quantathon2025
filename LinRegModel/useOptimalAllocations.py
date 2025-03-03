import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# 1. Load Data from OptimalAllocations and DailyStockBondFloats
# ------------------------------
optimal_path = "../data/OptimalAllocations_Optimized.csv"
daily_path = "../data/DailyStockBondFloats.csv"

optimal_df = pd.read_csv(optimal_path, parse_dates=["Date"])
daily_df   = pd.read_csv(daily_path, parse_dates=["Date"])

# Merge on Date (only keep days present in both)
merged_df = pd.merge(optimal_df, daily_df[["Date", "S&P500"]], on="Date", how="inner")

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
portfolio_value = initial_value
dynamic_values = []

# Loop over simulation days; each day rebalance according to given allocations.
for idx, row in sim_df.iterrows():
    stock_alloc = row["StockAllocation_Optimal"]
    bond_alloc  = row["BondAllocation_Optimal"]  # Assume this equals 1 - StockAllocation
    # Daily portfolio return: weighted average of stock return and bond daily return.
    daily_return = stock_alloc * row["StockReturn"] + bond_alloc * bond_daily_return
    portfolio_value *= (1 + daily_return)
    dynamic_values.append(portfolio_value)

sim_df["DynamicPortfolioValue"] = dynamic_values

# ------------------------------
# 5. Compute Buy-and-Hold Strategy for S&P 500
# ------------------------------
# Buy-and-hold: Invest $1000 on the first trading day and hold until end.
initial_sp500 = sim_df["S&P500"].iloc[0]
sim_df["BuyHoldValue"] = 1000 * (sim_df["S&P500"] / initial_sp500)

# ------------------------------
# 6. Plot the Results
# ------------------------------
plt.figure(figsize=(12, 6))
plt.plot(sim_df["Date"], sim_df["DynamicPortfolioValue"], label="Dynamic Rebalanced Strategy", color="blue")
plt.plot(sim_df["Date"], sim_df["BuyHoldValue"], label="Buy-and-Hold (S&P 500)", color="red", linestyle="--")
plt.xlabel("Date")
plt.ylabel("Portfolio Value ($)")
plt.title("Portfolio Value Over Time: Dynamic Strategy vs. Buy-and-Hold (2019-2022)")
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------
# 7. Print Final Portfolio Values
# ------------------------------
final_dynamic = sim_df["DynamicPortfolioValue"].iloc[-1]
final_buy_hold = sim_df["BuyHoldValue"].iloc[-1]
print(f"Final Dynamic Strategy Value at End of 2022: ${final_dynamic:.2f}")
print(f"Final Buy-and-Hold Value at End of 2022: ${final_buy_hold:.2f}")
