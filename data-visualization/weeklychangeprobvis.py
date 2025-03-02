import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV (Replace 'your_file.csv' with actual file path)
df = pd.read_csv("data/WeeklyChangeProbability.csv", parse_dates=["Date"])

# Sort data by date (ensuring chronological order)
df = df.sort_values(by="Date")

# Set plot style
sns.set(style="whitegrid", font_scale=1.2)

# 🎨 **Dual-Line Chart for `PrDec` & `PrInc`**
plt.figure(figsize=(12, 6))

plt.plot(df["Date"], df["PrDec"], label="Probability of -20% Decline", color="tab:orange", marker="o", linestyle="-", alpha=0.7)
plt.plot(df["Date"], df["PrInc"], label="Probability of +20% Increase", color="tab:blue", marker="o", linestyle="-", alpha=0.7)

# Titles & Labels
plt.xlabel("Date")
plt.ylabel("Probability")
plt.title("Predicted Probabilities of Stock Movement (12-Month Outlook)", fontsize=14, fontweight="bold")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

# **Save the Line Chart**
plt.savefig("stock_prediction_probabilities.png", dpi=300, bbox_inches="tight")

# 🎨 **Heatmap of Correlations**
plt.figure(figsize=(6, 5))
corr = df.drop(columns=["Date"]).corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)

plt.title("Probability Correlation Heatmap", fontsize=14, fontweight="bold")

# **Save the Heatmap**
plt.savefig("probability_correlation_heatmap.png", dpi=300, bbox_inches="tight")