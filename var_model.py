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

# Parametics VaR
    z_score = stats.norm.ppf(1-confidence_level)
    parametric_var = mean_return + (z_score * std_return)
    parametric_var_dollar = portfolio_value * abs(parametric_var)

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
        "monte_carlo_var_dollar": monte_carlo_var_dollar,
        "parametric_var": parametric_var,
        "parametric_var_dollar": parametric_var_dollar
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

def out_of_sample_backtest(ticker, train_start="2022-01-01", train_end="2024-06-01", test_start="2024-06-01", test_end="2026-01-01",
                           confidence_level=0.95, portfolio_value=10000):
    """
    Calucates VaR using a traingin period, then tests that threshold against a separte,
    later testing period the model has never seen.
    """
    # Training: Calculate VaR using the earliest period
    train_data = yf.download(ticker, start=train_start, end=train_end)
    train_prices = train_data["Close"]
    train_returns = train_prices.pct_change().dropna()
    historical_var = np.percentile(train_returns, 100-confidence_level*100)

    # Testing: Check the treshold against the later period
    test_data = yf.download(ticker, start=test_start, end=test_end)
    test_prices = test_data["Close"]
    test_returns=test_prices.pct_change().dropna()

    breaches = (test_returns < historical_var).sum()
    total_test_days = len(test_returns)
    actual_breach_rate = (breaches/total_test_days)*100
    expected_breach_rate = (1-confidence_level)*100

    # Print Out of Sample Backtest Results
    print(f"\n\tOUT OF SAMPLE BACKTEST: {ticker}")
    print(f"Training period: {train_start} to {train_end}")
    print(f"Testing period: {test_start} to {test_end}")
    print(f"VaR threshold (from training data): {historical_var*100:.2f}%")
    print(f"Total testing days: {total_test_days}")
    print(f"VaR breaches in testing period: {breaches.values[0]}")
    print(f"Expected breach rate: {expected_breach_rate:.1f}%")
    print(f"Acutal breach rate: {actual_breach_rate.values[0]:.2f}%")

    if actual_breach_rate.values[0] > expected_breach_rate*1.2:
        print("Result: Model UNDERESTIMATES risk out-of-sample")
    elif actual_breach_rate.values[0] < expected_breach_rate*0.8:
        print("Result: Model OVERESTIMATES risk out-of-sample")
    else: 
        print("Result: Model remains well-calibrated out-of-sample")

def stress_test(results, portfolio_value=10000):
    """
    Applies historical stress scenarios to estimate portfolio losses
    during extreme market events, comparing them to VaR estimates
    """
    ticker = results["ticker"]
    historical_var_dollar = results["historical_var_dollar"]
    returns = results["returns"]

    worst_day_return = returns.min().values[0]
    worst_day_date = returns.idxmin().values[0]

    # Define stress Scenario 
    scenarios = {
        f"Worst Single Day ({ticker} actual data)": worst_day_return,
        "Covid-19 Crash (Mar 16, 2020)": -0.12,
        "2008 Financial Crisis (Oct 15, 2008)": -0.09,
        "Black Monday 1987 (Oct 19, 1987)": -0.226
    }

    print(f"\n\tSTRESS TEST: {ticker}")
    print(f"Portfolio Value: ${portfolio_value:,.2f}")
    print(f"Historical VaR (95%): ${historical_var_dollar:,.2f}")
    print(f"Actual worst day in dataset: "
          f"{worst_day_return*100:.2f}% ({worst_day_date})")
    print(f"\n{'Scenario':<35}{'Shock':<10}{'Loss($)':<12}{'Var Exceeded By':<15}")
    print("-"*77)

    for scenario_name, shock in scenarios.items():
        stress_loss = portfolio_value*abs(shock)
        var_excess = stress_loss - historical_var_dollar
        var_excess_pct = (stress_loss/ historical_var_dollar - 1)*100

        print(f"{scenario_name:<35}{str(round(shock*100, 1)) + "%":<10}"
              f"${stress_loss:<11,.2f}"
              f"${var_excess:,.2f}({var_excess_pct:.0f}% worse)")

    # Print the insights
    print(f"\nKey insight: Under these historical scenarios, losses range from")
    print(f"{(portfolio_value * 0.08 / historical_var_dollar):.1f}x to "
          f"{(portfolio_value * 0.226/ historical_var_dollar):.1f}x larger")
    print(f"than the 95% VaR estimate of ${historical_var_dollar:,.2f}")
    print(f"This illustrates why VaR alone is insufficient for tail risk management.")


# Call the function for the stocks
aapl_results = calculate_var("AAPL")
tsla_results = calculate_var("TSLA")
backtest_var(aapl_results)
backtest_var(tsla_results)
out_of_sample_backtest("AAPL")
out_of_sample_backtest("TSLA")
stress_test(aapl_results)
stress_test(tsla_results)

# Print the results
print("\tRISK COMPARISON: AAPL vs TSLA")
print(f"{'Metric':<30}{'AAPL':<15}{'TSLA':<15}")
print(f"{'Std Deviation (Volatility)':<30}{aapl_results['std_return']:<15.4f}{tsla_results['std_return']:<15.4f}")
print(f"{'Historical VaR ($)':<30}{aapl_results['historical_var_dollar']:<15.2f}{tsla_results['historical_var_dollar']:<15.2f}")
print(f"{'Monte Carlo VaR ($)':<30}{aapl_results['monte_carlo_var_dollar']:<15.2f}{tsla_results['monte_carlo_var_dollar']:<15.2f}")
print(f"{'Parametric VaR ($)':<30}{aapl_results['parametric_var_dollar']:<15.2f}{tsla_results['parametric_var_dollar']:<15.2f}")

# Gap Calculation
aapl_gap = abs(aapl_results['historical_var_dollar'] - aapl_results['monte_carlo_var_dollar'])
tsla_gap = abs(tsla_results['historical_var_dollar'] - tsla_results['monte_carlo_var_dollar'])

# Visulaization
fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(14,6))
fig.suptitle("Value at Risk Comparison: AAPL vs TSLA (2022-2026)",
             fontsize=14, fontweight="bold")

# AAPL Returns Distribution Chart
ax1.hist(aapl_results["returns"], bins=50, color="steelblue", alpha=0.7, edgecolor="white", linewidth=0.5)
ax1.axvline(x=aapl_results["historical_var"], color="red", linewidth=1.5, linestyle="-.", label=f"Historical VaR: {abs(aapl_results['historical_var'])*100:.2f}%")
ax1.axvline(x=aapl_results["monte_carlo_var"], color="orange", linewidth=1.5, linestyle="--", label=f"Monte Carlo VaR: {abs(aapl_results['monte_carlo_var'])*100:.2f}%")
ax1.axvline(x=aapl_results["parametric_var"],color = "purple", linewidth=1.5, linestyle = ":", label=f"Parametric VaR: {abs(aapl_results['parametric_var'])*100:.2f}%")
ax1.set_title("AAPL Returns Distribution", fontsize=12, fontweight="bold")
ax1.set_xlabel("Daily Returns", fontsize=10)
ax1.set_ylabel("Frequency(Number of Days)", fontsize=10)
ax1.legend(fontsize=9)

# TSLA Returns Distribution Chart
ax2.hist(tsla_results["returns"], bins=50, color="seagreen", alpha=0.7, edgecolor="white", linewidth=0.5)
ax2.axvline(x=tsla_results["historical_var"], color="red", linewidth=1.5, linestyle="-.", label=f"Historical VaR: {abs(tsla_results['historical_var'])*100:.2f}%")
ax2.axvline(x=tsla_results["monte_carlo_var"], color="orange", linewidth=1.5, linestyle="--", label=f"Monte Carlo VaR: {abs(tsla_results['monte_carlo_var'])*100:.2f}%")
ax2.axvline(x=tsla_results["parametric_var"], color="purple", linewidth=1.5, linestyle= ":", label=f"Parametric VaR: {abs(tsla_results['parametric_var'])*100:.2f}%")
ax2.set_title("TSLA Returns Distribution", fontsize=12, fontweight="bold")
ax2.set_xlabel("Daily Returns", fontsize=10)
ax2.set_ylabel("Frequency(Number of Days)", fontsize=10)
ax2.legend(fontsize=9)

plt.tight_layout()
plt.savefig("var_comparison_aapl_tsla.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nChart saved as var_comparison_aapl_tsla.png")
