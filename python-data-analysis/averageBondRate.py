import pandas as pd

# Load dataset
file_path = "data/DailyStockBondFloats.csv"
df = pd.read_csv(file_path, parse_dates=["Date"])

# Filter data from 1/2/2019 to 12/30/2022
start_date = "2019-01-02"
end_date = "2022-12-30"
df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)].sort_values("Date").reset_index(drop=True)

# Initial investment
initial_investment = 1000  # Change as needed
current_balance = initial_investment

# Bond duration (13 weeks = 91 days)
bond_duration = 91

# Track reinvestments
investment_dates = []
balances = []

# Start buying bonds and rolling over investments
i = 0
while i < len(df):
    purchase_date = df.loc[i, "Date"]

    # Find the bond maturity date (91 days later)
    maturity_date = purchase_date + pd.Timedelta(days=bond_duration)

    # Find the closest available bond rate before maturity
    future_df = df[df["Date"] >= maturity_date]
    if future_df.empty:
        break  # No more data available to reinvest

    # Use the bond rate at the purchase date
    bond_rate = df.loc[i, "Bond Rate"] / 100  # Convert percentage to decimal

    # Calculate interest earned over 13 weeks (APR is for full year, so divide by 4)
    interest_earned = current_balance * (bond_rate / 4)

    # New balance after reinvesting
    current_balance += interest_earned

    # Store results
    investment_dates.append(maturity_date)
    balances.append(current_balance)

    # Move to the next bond purchase after maturity
    i = future_df.index[0]  # Jump to the next available date

# Print final amount
print(f"Final amount after reinvesting bonds from {start_date} to {end_date}: ${current_balance:.2f}")
