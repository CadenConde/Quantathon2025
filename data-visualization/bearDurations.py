import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, "data")
daily_stock_bond_path = os.path.join(data_dir, "DailyStockBondFloats.csv")

df = pd.read_csv(daily_stock_bond_path, parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# Define function to get bull and bear market segments
def get_segments(df):
    drop_threshold = 0.81   # Bull market ends if price < 80% of the running max
    rise_threshold = 1.20   # Bear market ends if price > 120% of the running min
    segments = []

    if len(df) == 0:
        print("Empty DataFrame")
        return segments

    state = "Bull"
    start_idx = 0
    current_max = df["S&P500"].iloc[0]
    current_max_idx = 0

    for i in range(1, len(df)):
        price = df["S&P500"].iloc[i]
        if state == "Bull":
            if price > current_max:
                current_max = price
                current_max_idx = i
            if price < current_max * drop_threshold:
                segments.append(("Bull", start_idx, current_max_idx))
                state = "Bear"
                start_idx = current_max_idx
                current_min = price
                current_min_idx = i
        elif state == "Bear":
            if price < current_min:
                current_min = price
                current_min_idx = i
            if price > current_min * rise_threshold:
                segments.append(("Bear", start_idx, current_min_idx))
                state = "Bull"
                start_idx = current_min_idx
                current_max = price
                current_max_idx = i

    segments.append((state, start_idx, len(df) - 1))
    return segments

# Get market segments
segments = get_segments(df)
print("Detected segments:", segments)

# Compute bear market durations
bear_market_durations = [df["Date"].iloc[e] - df["Date"].iloc[s] for phase, s, e in segments if phase == "Bear"]
bear_market_durations = [d.days for d in bear_market_durations]
print("Bear market durations (days):", bear_market_durations)

# Plot bear market duration as a horizontal boxplot
plt.figure(figsize=(8, 4))
sns.boxplot(x=bear_market_durations, color='tab:orange')
plt.xlabel('Duration (Days)')
plt.title('Bear Market Duration Boxplot')
plt.show()
