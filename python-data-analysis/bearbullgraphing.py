import pandas as pd
from bearbullsegments import segments, df
import matplotlib.pyplot as plt

print("Detected segments:", segments)

# Plot the S&P500 with segments colored by market phase.
plt.figure(figsize=(12, 6))
for phase, s, e in segments:
    color = "green" if phase == "Bull" else "red"
    plt.plot(df["Date"].iloc[s:e + 1], df["S&P500"].iloc[s:e + 1], color=color, linewidth=2)

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
