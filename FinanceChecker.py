import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from datetime import timedelta

# Fetch historical data for the stock
def fetch_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)
    if df.empty:
        raise ValueError("No data fetched. Check ticker symbol and date range.")
    return df

# Prepare data for the model with additional features
def prepare_data(df):
    df['Prediction'] = df['Close'].shift(-1)
    df['Moving_Avg'] = df['Close'].rolling(window=5).mean()  # Adding a moving average feature
    df = df.dropna()
    X = df[['Close', 'Moving_Avg']]
    y = df['Prediction']
    return X, y

# Define parameters
ticker = 'AAPL'  # Correct ticker symbol
start_date = '2019-01-01'
end_date = '2024-12-12'

# Fetch and prepare data
data = fetch_data(ticker, start_date, end_date)
print(data.head())  # Debugging: Check the first few rows of the fetched data

X, y = prepare_data(data)
print(f"Data shape after preparation: X={X.shape}, y={y.shape}")  # Debugging: Check data shape

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
print(f"Train shape: X_train={X_train.shape}, y_train={y_train.shape}")  # Debugging: Check train set shape
print(f"Test shape: X_test={X_test.shape}, y_test={y_test.shape}")  # Debugging: Check test set shape

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model using Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
predictions = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# Plot predictions vs actual values
plt.figure(figsize=(14, 7))
plt.plot(data.index[-len(y_test):], y_test, label='Actual Prices', color='blue')
plt.plot(data.index[-len(y_test):], predictions, label='Predicted Prices', color='red')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title(f'{ticker} Stock Price Prediction')
plt.legend()
plt.show()

# Forecast future prices using a smoother approach
def forecast_future_prices(model, last_known_price, num_days):
    future_dates = [data.index[-1] + timedelta(days=i) for i in range(1, num_days + 1)]
    future_prices = []
    current_price = last_known_price

    # Initialize a moving average to simulate a smoother forecast
    moving_avg_window = 5
    future_moving_avg = np.mean(data['Close'].tail(moving_avg_window))

    for _ in range(num_days):
        next_price = model.predict(scaler.transform([[current_price, future_moving_avg]]))[0]
        future_prices.append(next_price)
        current_price = next_price
        # Update moving average
        future_moving_avg = np.mean([future_moving_avg, next_price])

    return future_dates, future_prices

# Forecast for the next 30 days
future_dates, future_prices = forecast_future_prices(model, data['Close'].iloc[-1], 30)

# Plot future predictions
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['Close'], label='Historical Prices', color='blue')
plt.plot(future_dates, future_prices, label='Forecasted Prices', color='green')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title(f'{ticker} Stock Price Forecast')
plt.legend()
plt.show()
