import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf


# import data
df_dsb = pd.read_csv("../data/DailyStockBond.csv")
df_wcp = pd.read_csv("../data/WeeklyChangeProbability.csv")

# clean data


def str_to_float(val):
    return float(val.replace(",", ""))


def make_prexp(x, y):
    return x - y


df_dsb["Date"] = pd.to_datetime(df_dsb["Date"])
df_dsb["S&P500"] = df_dsb["S&P500"].apply(str_to_float)

df_wcp["Date"] = pd.to_datetime(df_wcp["Date"])
df_wcp["PrExp"] = df_wcp.apply(lambda row: make_prexp(row["PrInc"], row['PrDec']), axis=1)

df_dsb["Closest_PrExp"] = df_dsb["Date"].apply(lambda d: df_wcp.iloc[(
    (df_wcp["Date"] - d).dt.total_seconds() < 0).idxmax()]["PrExp"])

# plot data
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.scatter(df_wcp["Date"], df_wcp["PrInc"], label="PrInc", s=1, alpha=.5)
ax1.scatter(df_wcp["Date"], df_wcp["PrDec"], label="PrDec", s=1, alpha=.5)
ax1.scatter(df_wcp["Date"], df_wcp["PrExp"], label="PrExp", s=1, alpha=.5)
ax1.plot(df_wcp["Date"], np.zeros_like(df_wcp["Date"]), alpha=.5)
ax1.set_ylabel("Probability")
ax1.legend(loc='upper left')

colors = ["tab:blue" if pr_exp > 0 else "tab:orange" for pr_exp in df_dsb["Closest_PrExp"]]
for i in range(len(df_dsb) - 1):
    ax2.plot(df_dsb["Date"].iloc[i:i + 2], df_dsb["S&P500"].iloc[i:i + 2], color=colors[i])
ax2.set_ylabel("S&P500")

plt.show()
