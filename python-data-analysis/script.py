import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from bearbullsegments import get_segments


# import data
df_dsb = pd.read_csv("../data/DailyStockBond.csv", parse_dates=["Date"])
df_wcp = pd.read_csv("../data/WeeklyChangeProbability.csv")

# clean data


def str_to_float(val):
    return float(val.replace(",", ""))


def make_prexp(x, y):
    return x - y


df_dsb["S&P500"] = df_dsb["S&P500"].apply(str_to_float)
df_dsb = df_dsb.sort_values("Date").reset_index(drop=True)

df_wcp["Date"] = pd.to_datetime(df_wcp["Date"])
df_wcp["PrExp"] = df_wcp.apply(lambda row: make_prexp(row["PrInc"], row['PrDec']), axis=1)

df_dsb["Closest_PrExp"] = df_dsb["Date"].apply(lambda d: df_wcp.iloc[(
    (df_wcp["Date"] - d).dt.total_seconds() < 0).idxmax()]["PrExp"])

# jan12019 = df_dsb[df_dsb["Date"] == pd.to_datetime("2019-01-02")]["S&P500"].iloc[0]
# dec312022 = df_dsb[df_dsb["Date"] == pd.to_datetime("2022-12-30")]["S&P500"].iloc[0]
# print(dec312022 - jan12019)


segments = get_segments(df_dsb)
print(segments)

# plot data
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.scatter(df_wcp["Date"], df_wcp["PrInc"], label="PrInc", s=1, alpha=.5)
ax1.scatter(df_wcp["Date"], df_wcp["PrDec"], label="PrDec", s=1, alpha=.5)
ax1.scatter(df_wcp["Date"], df_wcp["PrExp"], label="PrExp", s=1, alpha=.5)
ax1.plot(df_wcp["Date"], np.zeros_like(df_wcp["Date"]), alpha=.5)
ax1.set_ylabel("Probability")
ax1.legend(loc='upper left')

for phase, s, e in segments:
    color = "tab:green" if phase == "Bull" else "tab:red"
    ax2.plot(df_dsb["Date"].iloc[s:e + 1], df_dsb["S&P500"].iloc[s:e + 1], color=color, linewidth=2)
ax2.set_ylabel("S&P500")

plt.show()
