# Author: Nafeesa Hassanin

# Import modules
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm

import time
start_time = time.time()

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

# T-Distribution Monte Carlo Var
    df_t, loc_t, scale_t = stats.t.fit(returns.squeeze())
    np.random.seed(42)
    simulated_returns_t = stats.t.rvs(df=df_t, loc=loc_t, scale=scale_t, size=num_simulations)
    t_dist_var = np.percentile(simulated_returns_t, 100 - confidence_level* 100)
    t_dist_var_dollar = portfolio_value * abs(t_dist_var)

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
        "df_t": df_t,
        "loc_t": loc_t,
        "scale_t": scale_t,
        "historical_var": historical_var,
        "historical_var_dollar": historical_var_dollar,
        "monte_carlo_var": monte_carlo_var,
        "monte_carlo_var_dollar": monte_carlo_var_dollar,
        "parametric_var": parametric_var,
        "parametric_var_dollar": parametric_var_dollar,
        "t_dist_var": t_dist_var,
        "t_dist_var_dollar": t_dist_var_dollar
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
    print(f"Actual breach rate: {actual_breach_rate.values[0]:.2f}%")

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

def plot_var_comparison(aapl_results, tsla_results, jpm_results):
    """
    Three histograms showing return distributions with VaR lines for each stock
    """
    fig, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(18,6))
    fig.suptitle("Return Distributions and VaR Thresholds (2022-2026)", 
                 fontsize=13, fontweight="bold")
    stocks= [
        (ax1, aapl_results, "royalblue", "AAPL"),
        (ax2, tsla_results, "seagreen", "TSLA"),
        (ax3, jpm_results, "goldenrod", "JPM")
    ]

    x_min = min(aapl_results["returns"].min().values[0],
                tsla_results["returns"].min().values[0],
                jpm_results["returns"].min().values[0])
    x_max = max(aapl_results["returns"].max().values[0],
                tsla_results["returns"].max().values[0],
                jpm_results["returns"].max().values[0])

    for ax, results, color, ticker in stocks:
        ax.hist(results["returns"], bins=50, color=color, alpha=0.6, edgecolor="white", linewidth= 0.5,
                density=True, label="Actual Returns")
        x_range= np.linspace(results["returns"].min().values[0],
                             results["returns"].max().values[0], 300)
        ax.set_xlim(x_min, x_max)
        normal_curve = stats.norm.pdf(x_range, results["mean_return"], results["std_return"])
        ax.plot(x_range, normal_curve, color="red", linewidth=2,
                label="Normal Distribution")
        ax.axvline(x=results["historical_var"], color="black", linewidth=1.2, linestyle="--",
                   label=f"Historical VaR: "
                   f"{abs(results['historical_var'])*100:.2f}%")
        ax.axvline(x=results["monte_carlo_var"], color="red", linewidth=1.2, linestyle=":",
                   label=f"Monte Carlo VaR: "
                   f"{abs(results['monte_carlo_var'])*100:.2f}%")
        
        ax.set_title(f"{ticker} Return Distribution", fontsize=11, fontweight="bold")
        ax.set_xlabel("Daily Return", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=7)
        ax.tick_params(axis="both", labelsize=7) 

    plt.tight_layout(rect=[0,0,1,0.95])
    plt.savefig("return_distributions.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: return_distributions.png")

def plot_rolling_volatility(aapl_results, tsla_results, jpm_results):
    """
    Line chart showing how each stock's volatitlity changed over time
    """       
    fig, ax = plt.subplots(figsize=(12,5))
    aapl_vol = aapl_results["returns"].rolling(window=30).std()*np.sqrt(252)
    tsla_vol = tsla_results["returns"].rolling(window=30).std()*np.sqrt(252)
    jpm_vol = jpm_results["returns"].rolling(window=30).std()*np.sqrt(252)

    ax.plot(aapl_vol.index, aapl_vol.values,
            color="royalblue", linewidth=1.2, label="AAPL")
    ax.plot(tsla_vol.index, tsla_vol.values,
            color="seagreen", linewidth=1.2, label="TSLA")
    ax.plot(jpm_vol.index, jpm_vol.values,
            color="goldenrod", linewidth=1.2, label="JPM")
    
    ax.set_title("Rolling 30-Day Annualized Volatitlity (2022-2026)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Annualized Volatility", fontsize=10)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    plt.savefig("rolling_volatility.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: rolling_volatility.png")

def plot_correlation_scatter(aapl_results, tsla_results, jpm_results):
    """
    Scatter plot showing pairwise return correlations between all three stocks
    """
    fig, ax = plt.subplots(figsize=(9,7))

    combined = pd.DataFrame({
        "AAPL": aapl_results["returns"].squeeze(),
        "TSLA": tsla_results["returns"].squeeze(),
        "JPM": jpm_results["returns"].squeeze()
    })

    corr = combined.corr()
    aapl_tsla = corr.loc["AAPL", "TSLA"]
    aapl_jpm = corr.loc["AAPL", "JPM"]
    tsla_jpm = corr.loc["TSLA", "JPM"]

    ax.scatter(combined["AAPL"], combined["TSLA"],
               alpha=0.3, color="royalblue", s=8,
               label=f"AAPL vs TSLA (r={aapl_tsla:.3f})")
    ax.scatter(combined["AAPL"], combined["JPM"],
               alpha=0.3, color="goldenrod", s=8,
               label=f"AAPL vs JPM (r={aapl_jpm:.3f})")
    ax.scatter(combined["TSLA"], combined["JPM"],
               alpha=0.3, color="seagreen", s=8,
               label=f"TSLA vs JPM (r={tsla_jpm:.3f})")
    ax.axhline(y=0, color="gray", linewidth=0.5)
    ax.axvline(x=0, color="gray", linewidth=0.5)
    ax.set_title("Pairwise Return Correlations - AAPL, TSLA, JPM (2022-2026)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Return (first stock in pair)", fontsize=10)
    ax.set_ylabel("Return (second stock in pair)", fontsize=10)
    ax.legend(fontsize=10)
    ax.tick_params(axis="both", labelsize=8)
    
    plt.tight_layout()
    plt.savefig("correlation_scatter.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: correlation_scatter.png")

def plot_var_bar_comparison(aapl_results, tsla_results, jpm_results):
    """
    Bar chart comparing all three VaR methods side by side for each stock
    """
    fig, axes = plt.subplots(1,3,figsize=(16,6))
    fig.suptitle("VaR Comparison by Method - $10,000 Portfolio (95% Confidence)",
                 fontsize=13, fontweight="bold")
    
    stocks = [
        (axes[0], aapl_results, "AAPL"),
        (axes[1], tsla_results, "TSLA"),
        (axes[2], jpm_results, "JPM")
    ]
    methods = ["Historical", "Monte Carlo\n(Normal)", "Parametric", "Monte Carlo\n(T-Dist)"]
    colors = ["forestgreen", "mediumblue", "firebrick", "indigo"]

    for ax, results, ticker in stocks:
        values = [
            results["historical_var_dollar"],
            results["monte_carlo_var_dollar"],
            results["parametric_var_dollar"],
            results["t_dist_var_dollar"]
        ]
    
        x = np.arange(len(methods))
        bars = ax.bar(x, values, width=0.6, color=colors, alpha=0.8,
                    edgecolor="white", linewidth=0.5)
        
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height()*0.5,
                    f"${value:.0f}",
                    ha="center", va="center",
                    fontsize=8, fontweight="bold", color="white")
        
        ax.set_title(ticker, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=8)
        ax.set_ylabel("Daily VaR ($)", fontsize=9)
        ax.tick_params(axis="y", labelsize=8)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)

    plt.tight_layout(rect=[0,0,1,0.95])
    plt.savefig("var_bar_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: var_bar_comparison.png")

def plot_distribution_comparison(aapl_results, tsla_results, jpm_results):
    """
    Compares normal distribution vs t-distribution overlay on actual returns for each stock
    """
    fig, axes = plt.subplots(1,3,figsize=(18,6))
    fig.suptitle("Normal vs T-Distribution Fit - AAPL, TSLA, JPM (2022-2026)",
                 fontsize=13, fontweight="bold")
    
    stocks = [
        (axes[0], aapl_results, "royalblue", "AAPL"),
        (axes[1], tsla_results, "seagreen", "TSLA"),
        (axes[2], jpm_results, "goldenrod", "JPM")
         ]

    for ax, results, color, ticker in stocks:
        returns = results["returns"]

        ax.hist(returns, bins=50, color=color, alpha=0.5,
                edgecolor="white", linewidth=0.5, density=True,
                label="Actual Returns")
        x_range = np.linspace(returns.min().values[0], returns.max().values[0], 300)
        normal_curve = stats.norm.pdf(x_range, results["mean_return"], results["std_return"])
        ax.plot(x_range, normal_curve, color="red", linewidth=2, linestyle="--",
               label="Normal Distribution")
        
        t_curve = stats.t.pdf(x_range, df=results["df_t"], loc=results["loc_t"], scale=results["scale_t"])
        ax.plot(x_range, t_curve, color="purple", linewidth=2, linestyle="-",
                label=f"T-Distribution "
                f"(df={results['df_t']:.2f})")
        
        ax.axvline(x=results["historical_var"], color="black", linewidth=1.2, linestyle="-.",
                   label=f"Historical VaR: "
                   f"{abs(results['historical_var'])*100:.2f}%")
        ax.axvline(x=results["monte_carlo_var"], color="red", linewidth=1.2, linestyle=":",
                   label=f"Normal VaR: "
                   f"{abs(results['monte_carlo_var'])*100:.2f}%")
        ax.axvline(x=results["t_dist_var"], color="purple", linewidth=1.2, linestyle=":",
                   label=f"T-Dist VaR: "
                   f"{abs(results['t_dist_var'])*100:.2f}%")
        ax.set_title(f"{ticker} - Normal vs T-Distribution", fontsize=11, fontweight="bold")
        ax.set_xlabel("Daily Return", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=6.5)
        ax.tick_params(axis="both", labelsize=7)

    plt.tight_layout(rect=[0,0,1,0.95])
    plt.savefig("distribution_comparison.png", dpi=150, bbox_inches ="tight")
    plt.show()
    print("Saved: distribution_comparison.png")

def print_summary(aapl_results, tsla_results, jpm_results):
    """
    Prints a clean summary of all key findings
    """
    print("\n"+"="*65)
    print("EXECUTIVE SUMMARY")
    print("="*65)

    print("\n\tRisk Profile (Historical VaR, $10,000 portoflio)")
    print(f" AAPL: ${aapl_results['historical_var_dollar']:.2f}"
          f" Volatility:{aapl_results['std_return']*100:.2f}")
    print(f" TSLA: ${tsla_results['historical_var_dollar']:.2f}"
          f" Volatility:{tsla_results['std_return']*100:.2f}")
    print(f" JPM: ${jpm_results['historical_var_dollar']:.2f}"
          f" Volatility:{jpm_results['std_return']*100:.2f}")
    
    print("\n\tT-Distribution Degrees of Freedom")
    print(f" AAPL: {aapl_results['df_t']:.2f} - "
          f"{'fat tails' if aapl_results['df_t'] < 10 else 'near-normal'}")
    print(f" TSLA: {tsla_results['df_t']:.2f} - "
          f"{'fat tails' if tsla_results['df_t'] < 10 else 'near-normal'}")
    print(f" JPM: {jpm_results['df_t']:.2f} - "
          f"{'fat tails' if jpm_results['df_t'] < 10 else 'near-normal'}")
    
    print("\n\tPairwise Correlations")
    combined = pd.DataFrame({
        "AAPL": aapl_results["returns"].squeeze(),
        "TSLA": tsla_results["returns"].squeeze(),
        "JPM": jpm_results["returns"].squeeze()
    })
    corr = combined.corr()
    print(f" AAPL vs TSLA: {corr.loc['AAPL', 'TSLA']:.3f}")
    print(f" AAPL vs JPM: {corr.loc['AAPL', 'JPM']:.3f}")
    print(f" TSLA vs JPM: {corr.loc['TSLA', 'JPM']:.3f}")

    print("\n\tKey Findings")
    print( "1. JPM has the lowest VaR but worst stress test results")
    print(" - low daily volaittly does not mean low tail risk")
    print(" 2. T-distribution VaR < Normal VaR at 95% confidence")
    print(" - fat tails reduce moderate tail probability whie increasing extreme tail probability")
    print(" 3. Tesla was hardest to model out-of-sample (3.54% breach reate vs 5% expected)")
    print(" - volatilate stocks are sensitive to which market regime they're trained on")
    print(" 4. All three methods convereged within $2 for each stock")
    print(" - cross-validates the simulation implementation")
    print("="*65)

# Call the function for the stocks
aapl_results = calculate_var("AAPL")
tsla_results = calculate_var("TSLA")
jpm_results = calculate_var("JPM")
backtest_var(aapl_results)
backtest_var(tsla_results)
backtest_var(jpm_results)
out_of_sample_backtest("AAPL")
out_of_sample_backtest("TSLA")
out_of_sample_backtest("JPM")
stress_test(aapl_results)
stress_test(tsla_results)
stress_test(jpm_results)
plot_var_comparison(aapl_results, tsla_results, jpm_results)
plot_rolling_volatility(aapl_results, tsla_results, jpm_results)
plot_correlation_scatter(aapl_results, tsla_results, jpm_results)
plot_var_bar_comparison(aapl_results, tsla_results, jpm_results)
plot_distribution_comparison(aapl_results, tsla_results, jpm_results)
print_summary(aapl_results, tsla_results, jpm_results)

end_time= time.time()
print(f"\nTotal runtime: {end_time - start_time:.2f} seconds")

