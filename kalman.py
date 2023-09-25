import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter


# Ticker symbol and date range
ticker_symbol = "AAPL"
start_date = "2021-01-01"
end_date = "2023-09-01"

# Download data from Yahoo Finance
df = yf.download(ticker_symbol, start=start_date, end=end_date)
df = df[['Adj Close']]  # Keep only the 'Adj Close' column
df.reset_index(inplace=True)

# Adding noise
observations = df['Adj Close'] + np.random.normal(0, 2, len(df))

# Create and fit a Kalman filter
kf = KalmanFilter(initial_state_mean=0, initial_state_covariance=1, observation_covariance=2, transition_covariance=0.01)
kf = kf.em(observations)
(filtered_state_means, _) = kf.filter(observations)

# Make a prediction for the next state
(next_state_mean, next_state_covariance) = kf.filter_update(filtered_state_means[-1:], np.array([[1]]))

# Plot original data and filter 
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Adj Close'], label='Original Data', color='blue', alpha=0.5)
plt.plot(df['Date'], filtered_state_means, label='Kalman Filter', color='red')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Stock Price Prediction with Kalman Filter')
plt.legend()
plt.grid(True)
plt.show()

# Print predicted next state
print("Predicted next state (Price):", next_state_mean[0])
