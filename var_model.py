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
