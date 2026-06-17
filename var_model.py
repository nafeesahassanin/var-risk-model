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

# Historical VaR Calculation
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

# Monte Carlo Var Caclulation
# Extract mean and standard deviation of returns
mean_return = returns.mean().values[0]
std_return = returns.std().values[0]

# Set a random seed for reproducibility
np.random.seed(42)

# Generate 10,000 simulated daily returns
num_simulations = 10000
simulated_returns = np.random.normal(mean_return, std_return, num_simulations)

# Find the 5th percentile of simulated returns
monte_carlo_var = np.percentile(simulated_returns, 100 - confidence_level * 100)

monte_carlo_var_dollar = portfolio_value * abs(monte_carlo_var)

# Print the results
print("\n MONTE CARLO VAR")
print(f"Number of Simulations: {num_simulations:,}")
print(f"Mean daily return used: {mean_return:.4f}")
print(f"Standard deviation used: {std_return:.4f}")
print(f"Monte Carlo VaR (daily): {monte_carlo_var:.4f} ({monte_carlo_var * 100:.2f}%)")
print(f"Monte Carlo VaR in dollars: ${monte_carlo_var_dollar:,.2f}")
print(f"Interpretation: Simulated model suggests losses won't exceed ${monte_carlo_var_dollar:,.2f} on 95% of days")

# Comparison
print("\n COMPARISON")
print(f"Historical VaR: ${historical_var_dollar:,.2f} ({historical_var *100:.2f}%)")
print(f"Monte Carlo VaR: ${monte_carlo_var_dollar:,.2f} ({monte_carlo_var * 100:.2f}%)")
print(f"Difference: ${abs(historical_var_dollar - monte_carlo_var_dollar):,.2f}")

if historical_var_dollar > monte_carlo_var_dollar:
    print("Insight: Historical VaR is higher than Monte Carlo VaR")
    print("This suggests real returns have fatter tails than normal")
    print("distribution predicts - extreme losses happen more often")
    print("in reality than the model assumes.")
else:
    print("Insight: Monte Carlo VaR is higher than Historical VaR")
    print("This suggesrs the normal distribution is overestimating")
    print("tail risk compared to what actually occurred in this period.")

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f"Value at Risk Analysis - {ticker} (2022-2026)", size=14, fontweight="bold")

# Historical Returns Distribution
ax1.hist(returns, bins=50, color="steelblue", alpha=0.7, edgecolor="white", linewidth=0.5)
ax1.axvline(x=historical_var, color="red", linewidth=2, linestyle="--", label=f"Histroical VaR: {historical_var*100:.2f}%")
ax1.set_title("Historical Returns Distributions", fontsize=12, fontweight="bold")
ax1.set_xlabel("Daily Return", fontsize=10)
ax1.set_ylabel("Frequency (Number of Days)", fontsize=10)
ax1.legend(fontsize=9)
ax1.annotate(f"${historical_var_dollar:,.2f} at risk \n(95% confidence)",
             xy=(historical_var, ax1.get_ylim()[1]*0.3),
             xytext=(historical_var - 0.06, ax1.get_ylim()[1]*0.6),
             fontsize=8, color="red",
             arrowprops=dict(arrowstyle="->", color="red"))

# Monte Carlo Simulated Returns Distribution
ax2.hist(simulated_returns, bins=50, color="seagreen", alpha=0.7, edgecolor="white", linewidth=0.5)
ax2.axvline(x=monte_carlo_var, color="red", linewidth=2, linestyle="--", label=f"Monte Carlo VaR: {monte_carlo_var*100:.2f}%")
ax2.set_title("Monte Carlo Simulated Returns Distribution", fontsize=12, fontweight="bold")
ax2.set_xlabel("Simulated Daily Return", fontsize=10)
ax2.set_ylabel("Frequency (Number of Simulated Days)", fontsize=10)
ax2.legend(fontsize=9)
ax2.annotate(f"${monte_carlo_var_dollar:,.2f} at risk \n(95% confidence)",
             xy=(monte_carlo_var, ax2.get_ylim()[1]*0.5),
             xytext=(monte_carlo_var - 0.03, ax2.get_ylim()[1]*0.7),
             fontsize=8, color="red",
             arrowprops=dict(arrowstyle="->", color="red"))

plt.tight_layout()
plt.savefig("var_analysis.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nChart saved as var_analysis.png")