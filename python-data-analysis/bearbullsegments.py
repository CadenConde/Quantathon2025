import pandas as pd
import os

# Load dataset (adjust path as needed)
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, "data")
daily_stock_bond_path = os.path.join(data_dir, "DailyStockBondFloats.csv")

df = pd.read_csv(daily_stock_bond_path, parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)


def get_segments(df):
    # Define thresholds for 20% change
    drop_threshold = 0.81   # In a bull market, if price < 80% of the running maximum, the bull phase ends.
    rise_threshold = 1.20   # In a bear market, if price > 120% of the running minimum, the bear phase ends.

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
        return segments
