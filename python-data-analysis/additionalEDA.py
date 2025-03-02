import pandas as pd
import matplotlib.pyplot as plt

# Load data
daily_data = pd.read_csv("../data/DailyStockBondFloats.csv")
weekly_probs = pd.read_csv("../data/WeeklyChangeProbability.csv")

# Convert date columns to datetime
daily_data['Date'] = pd.to_datetime(daily_data['Date'])
weekly_probs['Date'] = pd.to_datetime(weekly_probs['Date'])

# Sort by date
daily_data = daily_data.sort_values('Date')
weekly_probs = weekly_probs.sort_values('Date')

# Rolling Statistics (50-day moving average & standard deviation)
daily_data['Rolling_Mean'] = daily_data['S&P500'].rolling(window=50).mean()
daily_data['Rolling_Std'] = daily_data['S&P500'].rolling(window=50).std()

# Plot rolling statistics
plt.figure(figsize=(12, 6))
plt.plot(daily_data['Date'], daily_data['S&P500'], label='S&P 500', alpha=0.5)
plt.plot(daily_data['Date'], daily_data['Rolling_Mean'], label='50-day MA', color='tab:orange')
plt.fill_between(daily_data['Date'], 
                 daily_data['Rolling_Mean'] - daily_data['Rolling_Std'],
                 daily_data['Rolling_Mean'] + daily_data['Rolling_Std'], 
                 color='gray', alpha=0.3, label='Rolling Std Dev')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('S&P 500 Rolling Statistics')
plt.legend()
plt.show()

# Correlation Analysis
merged_data = pd.merge(daily_data, weekly_probs, on='Date', how='inner')
correlation = merged_data[['S&P500', 'PrInc', 'PrDec']].corr()
print("Correlation Matrix:\n", correlation)

# Bear Market Length Distribution
bear_market_durations = []
peak = daily_data['S&P500'].iloc[0]
bear_start = None

for i in range(1, len(daily_data)):
    if daily_data['S&P500'].iloc[i] > peak:
        peak = daily_data['S&P500'].iloc[i]  # Update peak if a new high is reached
    if daily_data['S&P500'].iloc[i] <= peak * 0.8 and bear_start is None:
        bear_start = daily_data['Date'].iloc[i]  # Start of bear market
    if daily_data['S&P500'].iloc[i] >= peak * 0.9 and bear_start is not None:
        duration = (daily_data['Date'].iloc[i] - bear_start).days
        bear_market_durations.append(duration)
        print(f"Bear market duration: {duration} days")
        bear_start = None  # Reset after recovery

print("All bear market durations:", bear_market_durations)

plt.figure(figsize=(8, 5))
plt.hist(bear_market_durations, bins=10, edgecolor='black', alpha=0.7, color='tab:blue')
plt.xlabel('Duration (Days)')
plt.ylabel('Frequency')
plt.title('Bear Market Duration Distribution')
plt.show()