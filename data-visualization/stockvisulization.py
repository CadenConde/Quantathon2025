import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = "C:\Users\rlee0\OneDrive - The Ohio State University\AppsOSU\Quant2025\data\WeeklyChangeProbability.csv"
df = pd.read_csv(file_path)

# Convert Date column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Sort by date in ascending order
df = df.sort_values(by="Date")

# Convert probabilities to percentages
df['PrDec'] *= 100
df['PrInc'] *= 100

# Plot probabilities over time as percentages
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['PrDec'], label='Probability of 20%+ Decrease', color='red')
plt.plot(df['Date'], df['PrInc'], label='Probability of 20%+ Increase', color='green')

# Formatting the plot
plt.xlabel("Date")
plt.ylabel("Probability (%)")
plt.title("Market-Based Probabilities of 20%+ Change in S&P 500")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
