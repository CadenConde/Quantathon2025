import tensorflow as tf
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

# Load dataset (2007-2018)
file_path = "data/DailyStockBondWithInterpolatedProbabilities.csv"
df = pd.read_csv(file_path)
df["Date"] = pd.to_datetime(df["Date"])
train_df = df[(df["Date"] >= "2007-01-01") & (df["Date"] <= "2018-12-31")].copy()

# Compute daily returns
train_df["S&P500 Return"] = train_df["S&P500"].pct_change().fillna(0)
train_df["Bond Daily Return"] = train_df["Bond Rate"] / 25200  # Convert bond rate to daily return

# Define input features and target
X_train = train_df[["PrInc", "PrDec", "Bond Rate"]].values
y_train = train_df["S&P500 Return"].values

# Scale inputs
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

# List of epsilon values to test
epsilon_values = [1e-6, 1e-5, 1e-4, 1e-3]
results = []

# Train model for each epsilon value
for epsilon in epsilon_values:
    print(f"\n🔹 Testing Epsilon: {epsilon}")  # Debugging Step 1

    # Custom Sharpe Ratio Loss Function
    def sharpe_loss(y_true, y_pred):
        portfolio_return = y_pred * y_true
        portfolio_std = tf.keras.backend.std(portfolio_return)
        
        # Clip standard deviation to prevent division by near-zero values
        portfolio_std = tf.clip_by_value(portfolio_std, 1e-3, np.inf)

        # Add penalty for too much stock exposure
        penalty = tf.reduce_mean(tf.square(tf.clip_by_value(y_pred, 1.1, np.inf))) * 10  

        loss = -tf.reduce_mean(portfolio_return) / (portfolio_std + epsilon) + penalty
        return loss

    # Build Neural Network Model with Dropout to Reduce Overfitting
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(3,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),  # Dropout to prevent overfitting
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation=None)  # Use linear activation to allow flexible allocations
    ])

    # Compile Model with Different Optimizer
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss=sharpe_loss, metrics=['mae'])

    print(f"⚙️ Training model for Epsilon: {epsilon}...")  # Debugging Step 2
    history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=16, verbose=1) 

    final_loss = history.history['loss'][-1]
    print(f"✅ Finished Training. Final Loss for Epsilon {epsilon}: {final_loss}")  # Debugging Step 3

    # Simulate Portfolio Performance Using Model Predictions
    print(f"📊 Running Portfolio Simulation for Epsilon: {epsilon}...")  # Debugging Step 4

    portfolio_value = 10000  # Start with $10,000
    
    # **Vectorized Prediction for Speed**
    allocations = np.clip(model.predict(X_train_scaled, verbose=0).flatten(), 0, 1)  # Keep between 0% and 100%

    for i in range(len(y_train)):
        allocation = allocations[i]  # Get model-predicted allocation

        # Cap Maximum Growth (Prevent Exponential Growth)
        max_daily_growth = 1.05  # Max 5% daily growth
        portfolio_value *= min(1 + allocation * y_train[i], max_daily_growth)

    print(f"✅ Finished Portfolio Simulation for Epsilon: {epsilon}")  # Debugging Step 5

    # Store results
    results.append((epsilon, portfolio_value))

# Find the Best Epsilon Value
best_epsilon, best_value = max(results, key=lambda x: x[1])

# Print Final Results
print("\n🔍 Performance of Different Epsilon Values:")
for epsilon, value in results:
    print(f"✅ Epsilon = {epsilon}: Final Portfolio Value = ${value:,.2f}")

print(f"\n🏆 Best Epsilon: {best_epsilon} with Final Portfolio Value = ${best_value:,.2f}")