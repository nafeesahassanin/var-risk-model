# Author: Nafeesa Hassanin

# Import modules
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Choose stock
ticker = "AAPL"

# Download the data
data = yf.download(ticker, start="2022-01-01", end="2026-01-01")

prices = data["Close"]

returns = prices.pct_change().dropna()

# Print the statistics
print(f"Stock: {ticker}")
print(f"Number of trading days: {len(returns)}")
print(f"Average daily return: {returns.mean().values[0]:.4f}")
print(f"Standard deviation:{returns.std().values[0]:.4f}")
print(f"Worst single day: {returns.min().values[0]:.4f}")
print(f"Best single day: {returns.max().values[0]:.4f}")

# Define confidence level and portfolio size
confidence_level = 0.95
portfolio_value = 10000

# Find the 5th percential of historical returns
historical_var = np.percentile(returns, 100 -  confidence_level * 100)

historical_var_dollar = portfolio_value * abs(historical_var)

# Print the results
print("\n HISTORICAL VAR")
print(f"Confidence Level: {confidence_level * 100:.0f}%")
print(f"Portfolio Value: ${portfolio_value:,.0f}")
print(f"Historical VaR (daily): {historical_var:.4f} ({historical_var * 100:.2f}%)")
print(f"Historical VaR in dollars: ${historical_var_dollar:,.2f}")
print(f"Interpretation: On 95% of days, losses won't exceed ${historical_var_dollar:,.2f}")
