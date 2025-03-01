import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf


# import data
df_dsb = pd.read_csv("../data/DailyStockBond.csv")
df_wcp = pd.read_csv("../data/WeeklyChangeProbability.csv")

# clean data
def str_to_float(val):
    return float(val.replace(",",""))

df_dsb["Date"] = pd.to_datetime(df_dsb["Date"])
df_dsb["S&P500"] = df_dsb["S&P500"].apply(str_to_float)

# plot data
plt.plot(df_dsb["Date"], df_dsb["S&P500"])
plt.show()
