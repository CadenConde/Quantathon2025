import pandas as pd
import matplotlib.pyplot as plt
import os

# Load dataset (adjust path as needed)
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, "data")
daily_stock_bond_path = os.path.join(data_dir, "DailyStockBondFloats.csv")

df = pd.read_csv(daily_stock_bond_path, parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# Define thresholds for 20% change
drop_threshold = 0.8   # In a bull market, if price < 80% of the running maximum, the bull phase ends.
rise_threshold = 1.2   # In a bear market, if price > 120% of the running minimum, the bear phase ends.

# We'll use a state machine approach.
segments = []  # List of tuples: (phase, start_idx, end_idx)

if len(df) == 0:
    print("Empty DataFrame")
else:
    # Assume market starts in a bull phase.
    state = "Bull"
    start_idx = 0
    # For bull market, track the running maximum.
    current_max = df["S&P500"].iloc[0]
    current_max_idx = 0

    # Loop over all prices
    for i in range(1, len(df)):
        price = df["S&P500"].iloc[i]
        if state == "Bull":
            # Update the running maximum if a new high is reached.
            if price > current_max:
                current_max = price
                current_max_idx = i
            # If the price falls below 80% of the running max, end the bull market.
            if price < current_max * drop_threshold:
                # Record bull market from start_idx to the index of the running max.
                segments.append(("Bull", start_idx, current_max_idx))
                # Switch state to Bear; bear market begins at the peak.
                state = "Bear"
                start_idx = current_max_idx
                # Initialize bear tracking with current price.
                current_min = price
                current_min_idx = i
        elif state == "Bear":
            # Update the running minimum if a new low is reached.
            if price < current_min:
                current_min = price
                current_min_idx = i
            # If the price rises above 120% of the running minimum, end the bear market.
            if price > current_min * rise_threshold:
                # Record bear market from start_idx to the index of the running minimum.
                segments.append(("Bear", start_idx, current_min_idx))
                # Switch state to Bull; bull market begins at the trough.
                state = "Bull"
                start_idx = current_min_idx
                # Initialize bull tracking with current price.
                current_max = price
                current_max_idx = i

    # Record the final segment from start_idx to the end of data.
    segments.append((state, start_idx, len(df) - 1))

# print("Detected segments:", segments)

# Plot the S&P500 with segments colored by market phase.
plt.figure(figsize=(12, 6))
for phase, s, e in segments:
    color = "green" if phase == "Bull" else "red"
    plt.plot(df["Date"].iloc[s:e+1], df["S&P500"].iloc[s:e+1], color=color, linewidth=2)

plt.xlim(pd.Timestamp("2008-01-01"), pd.Timestamp("2023-12-31"))
plt.ylim(700, 7000)
plt.xlabel("Date")
plt.ylabel("S&P 500 Index")
plt.title("S&P 500 Market Phases\n(Bull=Green, Bear=Red)")

# Add legend with dummy plots.
plt.plot([], [], color="green", label="Bull Market")
plt.plot([], [], color="red", label="Bear Market")
plt.legend()

plt.show()
