# Author: Nafeesa Hassanin

# Import modules
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Create function to calculate VaR using both historical and Monte Carlo methods
def calculate_var(ticker, start_date="2022-01-01", end_date="2026-01-01", 
                  confidence_level=0.95, portfolio_value=10000):
# Download the data
    data = yf.download(ticker, start=start_date, end=end_date)
    prices = data["Close"]
    returns = prices.pct_change().dropna()
#Historical VaR Calculation
    historical_var = np.percentile(returns, 100 - confidence_level * 100)
    historical_var_dollar = portfolio_value * abs(historical_var)
# Monte Carlo VaR Calculation
    mean_return = returns.mean().values[0]
    std_return = returns.std().values[0]
    np.random.seed(42)
    num_simulations = 10000
    simulated_returns = np.random.normal(mean_return, std_return, num_simulations)
    monte_carlo_var = np.percentile(simulated_returns, 100 - confidence_level * 100)
    monte_carlo_var_dollar = portfolio_value * abs(monte_carlo_var)
# Create a dictionary to store the results
    results = {
        "ticker": ticker,
        "returns": returns,
        "simulated_returns": simulated_returns,
        "mean_return": mean_return,
        "std_return": std_return,
        "historical_var": historical_var,
        "historical_var_dollar": historical_var_dollar,
        "monte_carlo_var": monte_carlo_var,
        "monte_carlo_var_dollar": monte_carlo_var_dollar
    }
    return results

def backtest_var(results, confidence_level=0.95):
    """
    Backtests a VaR model by checking how often actual losses exceed the predicted VaR threshold.
    """
    returns = results["returns"]
    historical_var = results["historical_var"]
    # Count how many days had a loss worse than the VaR threshold
    breaches = (returns < historical_var).sum()
    total_days = len(returns)
    # What we observed
    actual_breach_rate = (breaches / total_days)*100
    # What we expected
    expected_breach_rate = (1 - confidence_level)*100

    print(f"\nBACKTEST: {results['ticker']}")
    print(f"Total trading days: {total_days}")
    print(f"VaR breaches: {breaches.values[0]}")
    print(f"Expected breach rate: {expected_breach_rate:.1f}%")
    print(f"Actual breach rate: {actual_breach_rate.values[0]:.2f}%")

    if actual_breach_rate.values[0] > expected_breach_rate * 1.2:
        print("Result: Model UNDERESTIMATES risk (breaches happened more than expected)")
    elif actual_breach_rate.values[0]< expected_breach_rate * 0.8:
        print("Results: Model OVERESTIMATES risk (breaches happened less than expected)")
    else: 
        print("Result: Model is well-calibrated (breach rate close to expected)")

# Call the function for the stocks
aapl_results = calculate_var("AAPL")
tsla_results = calculate_var("TSLA")
backtest_var(aapl_results)
backtest_var(tsla_results)

# Print the results
print("\tRISK COMPARISON: AAPL vs TSLA")
print(f"{'Metric':<30}{'AAPL':<15}{'TSLA':<15}")
print(f"{'Std Deviation (Volatility)':<30}{aapl_results['std_return']:<15.4f}{tsla_results['std_return']:<15.4f}")
print(f"{'Historical VaR ($)':<30}{aapl_results['historical_var_dollar']:<15.2f}{tsla_results['historical_var_dollar']:<15.2f}")
print(f"{'Monte Carlo VaR ($)':<30}{aapl_results['monte_carlo_var_dollar']:<15.2f}{tsla_results['monte_carlo_var_dollar']:<15.2f}")

# Gap Calculation
aapl_gap = abs(aapl_results['historical_var_dollar'] - aapl_results['monte_carlo_var_dollar'])
tsla_gap = abs(tsla_results['historical_var_dollar'] - tsla_results['monte_carlo_var_dollar'])

# Visulaization
fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(14,6))
fig.suptitle("Value at Risk Comparison: AAPL vs TSLA (2022-2026)",
             fontsize=14, fontweight="bold")

# AAPL Returns Distribution Chart
ax1.hist(aapl_results["returns"], bins=50, color="steelblue", alpha=0.7, edgecolor="white", linewidth=0.5)
ax1.axvline(x=aapl_results["historical_var"], color="red", linewidth=2, linestyle="--", label=f"Historical VaR: {aapl_results['historical_var']*100:.2f}%")
ax1.axvline(x=aapl_results["monte_carlo_var"], color="orange", linewidth=2, linestyle="--", label=f"Monte Carlo VaR: {aapl_results['monte_carlo_var']*100:.2f}%")
ax1.set_title("AAPL Returns Distribution", fontsize=12, fontweight="bold")
ax1.set_xlabel("Daily Returns", fontsize=10)
ax1.set_ylabel("Frequency(Number of Days)", fontsize=10)
ax1.legend(fontsize=9)

# TSLA Returns Distribution Chart
ax2.hist(tsla_results["returns"], bins=50, color="seagreen", alpha=0.7, edgecolor="white", linewidth=0.5)
ax2.axvline(x=tsla_results["historical_var"], color="red", linewidth=2, linestyle="--", label=f"Historical VaR: {tsla_results['historical_var']*100:.2f}%")
ax2.axvline(x=tsla_results["monte_carlo_var"], color="orange", linewidth=2, linestyle="--", label=f"Monte Carlo VaR: {tsla_results['monte_carlo_var']*100:.2f}%")
ax2.set_title("TSLA Returns Distribution", fontsize=12, fontweight="bold")
ax2.set_xlabel("Daily Returns", fontsize=10)
ax2.set_ylabel("Frequency(Number of Days)", fontsize=10)
ax2.legend(fontsize=9)

plt.tight_layout()
plt.savefig("var_comparison_aapl_tsla.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nChart saved as var_comparison_aapl_tesla.png")