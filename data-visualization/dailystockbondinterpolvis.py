import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV (Replace 'your_file.csv' with the actual file path)
df = pd.read_csv("data/DailyStockBondWithInterpolatedProbabilities.csv", parse_dates=["Date"])

# Set plot style
sns.set(style="whitegrid", font_scale=1.2)

# 🎨 **Dual-Axis Line Plot (S&P 500 vs. Bond Rate)**
fig, ax1 = plt.subplots(figsize=(12, 6))

# Primary axis: S&P 500
ax1.set_xlabel("Year")
ax1.set_ylabel("S&P 500 Index", color="tab:blue")
ax1.plot(df["Date"], df["S&P500"], label="S&P 500", color="tab:blue", alpha=0.7)
ax1.tick_params(axis="y", labelcolor="tab:blue")

# Secondary axis: Bond Rate
ax2 = ax1.twinx()
ax2.set_ylabel("Bond Rate (%)", color="tab:orange")
ax2.plot(df["Date"], df["Bond Rate"], label="Bond Rate", color="tab:orange", linestyle="dashed", alpha=0.7)
ax2.tick_params(axis="y", labelcolor="tab:orange")

# Titles & Legend
plt.title("S&P 500 vs. Bond Rate (2007-Present)", fontsize=14, fontweight="bold")
fig.tight_layout()

# Save plot
plt.savefig("sp500_vs_bond_rate.png", dpi=300, bbox_inches="tight")

# 🎨 **Heatmap of Correlations**
plt.figure(figsize=(8, 5))
corr = df.drop(columns=["Date"]).corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)

plt.title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")

plt.savefig("correlation_heatmap.png", dpi=300, bbox_inches="tight")