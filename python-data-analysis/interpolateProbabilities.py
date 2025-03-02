import pandas as pd

# Load dataset
file_path = "../data/DailyStockBondWithProbabilities.csv"
df = pd.read_csv(file_path, parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# Apply linear interpolation to fill NaN values
df["PrDec"] = df["PrDec"].interpolate()
df["PrInc"] = df["PrInc"].interpolate()

# Save the completed dataset
output_path = "../data/DailyStockBondWithInterpolatedProbabilities.csv"
df.to_csv(output_path, index=False)

print(f"Saved interpolated dataset: {output_path}")
