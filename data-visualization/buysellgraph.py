import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the CSV file
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
data_dir = os.path.join(base_dir, "data") 

daily_stock_bond_path = os.path.join(data_dir, "DailyStockBondFloats.csv")

df = pd.read_csv(daily_stock_bond_path, parse_dates=["Date"])

# Sort by date in case the data is unordered
df = df.sort_values("Date")

# Calculate 12-month forward return
df["Future"] = df["S&P500"].shift(-252)  # Assuming 252 trading days in a year
df["Return"] = (df["Future"] - df["S&P500"]) / df["S&P500"]

# Define market conditions
df["Color"] = "lightgray"  # Default color
df.loc[df["Return"] >= 0.2, "Color"] = "green"  # Bull market
df.loc[df["Return"] <= -0.2, "Color"] = "red"   # Bear market

# Plot the S&P 500 with appropriate colors
plt.figure(figsize=(12, 6))

# Plot each segment with the correct color
for i in range(len(df) - 1):
    plt.plot(df["Date"].iloc[i:i+2], df["S&P500"].iloc[i:i+2], color=df["Color"].iloc[i], linewidth=1)

# Set axis limits
plt.xlim(pd.Timestamp("2008-01-01"), pd.Timestamp("2023-12-31"))
plt.ylim(700, 7000)

# Add legend
plt.plot([], [], color="green", label="Buy")  # Empty plot for legend
plt.plot([], [], color="red", label="Sell")    # Empty plot for legend
plt.plot([], [], color="lightgray", label="Static")   # Empty plot for legend
plt.legend()

plt.xlabel("Date")
plt.ylabel("S&P 500 Index")
plt.title("S&P 500 with Bull (Green) and Bear (Red) Markets")
plt.show()