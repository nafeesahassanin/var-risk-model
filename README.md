# Value at Risk (VaR) Model — AAPL vs TSLA

## Overview
This project builds a Value at Risk (VaR) model in Python to estimate 
and compare the daily risk of a $10,000 investment in two stocks with 
very different volatility profiles: Apple (AAPL) and Tesla (TSLA). 
The model uses four years of real market data (2022–2026), applies three 
estimation methods — Historical Simulation, Monte Carlo Simulation, and 
Parametric (Variance-Covariance) — and validates results through both 
in-sample and out-of-sample backtesting.

## What is Value at Risk?
A 95% confidence VaR of $287 means: on 95% of trading days, losses will 
not exceed $287. On the worst 5% of days — roughly 12–13 trading days 
per year — losses are expected to exceed that threshold. 
## Methodology

### 1. Historical VaR
Uses real daily returns from each stock's full price history to find the 
actual 5th percentile of historical losses. No distributional assumptions 
are made — this method reflects what genuinely happened in the market 
across the full sample period.

### 2. Monte Carlo VaR
Generates 10,000 simulated daily returns drawn from a normal distribution 
parameterized by each stock's real historical mean return and standard 
deviation. The 5th percentile of the simulated distribution becomes the 
risk estimate. This method assumes returns are normally distributed — an 
assumption whose limitations become visible in the comparison below.

### 3. Parametric VaR (Variance-Covariance)
Uses a direct mathematical formula rather than simulation. Applies a 
known z-score constant (1.645 for 95% confidence) derived from the 
properties of the normal distribution to calculate VaR analytically 
in a single step. Compared to Monte Carlo as a cross-validation check — 
both methods use the same normal distribution assumption, so their 
results should converge, and they did (within $2 for both stocks), 
confirming the Monte Carlo simulation was implemented correctly.

### 4. In-Sample Backtesting
Tests whether the Historical VaR threshold, calculated from the full 
dataset, accurately predicted breach frequency against that same dataset. 
This is a standard first validation step, though it is inherently 
somewhat circular since the threshold is defined as the 5th percentile 
of the same data being tested.

### 5. Out-of-Sample Backtesting
Addresses the circularity of in-sample testing by splitting the dataset 
into two distinct periods — a training window (2022–2024) used to 
calculate the VaR threshold, and a testing window (2024–2026) used to 
validate it against data the model has never seen. 

### Stress Test Results

Stress testing applies historical crisis scenarios to estimate losses 
during extreme market events, directly comparing them to VaR estimates 
to quantify what VaR alone misses.

#### AAPL Stress Test (Historical VaR baseline: $287.39)
| Scenario | Shock | Loss ($) | vs. VaR |
|---|---|---|---|
| Worst Single Day (Apr 3, 2025) | -9.2% | $924.56 | 222% worse |
| COVID-19 Crash (Mar 16, 2020) | -12.0% | $1,200.00 | 318% worse |
| 2008 Financial Crisis (Oct 15, 2008) | -9.0% | $900.00 | 213% worse |
| Black Monday (Oct 19, 1987) | -22.6% | $2,260.00 | 686% worse |

#### TSLA Stress Test (Historical VaR baseline: $611.51)
| Scenario | Shock | Loss ($) | vs. VaR |
|---|---|---|---|
| Worst Single Day (Mar 10, 2025) | -15.4% | $1,542.62 | 152% worse |
| COVID-19 Crash (Mar 16, 2020) | -12.0% | $1,200.00 | 96% worse |
| 2008 Financial Crisis (Oct 15, 2008) | -9.0% | $900.00 | 47% worse |
| Black Monday (Oct 19, 1987) | -22.6% | $2,260.00 | 270% worse |

#### Key Stress Test Insights

**VaR dramatically underestimates tail risk for both stocks.**
For Apple, crisis scenarios produced losses ranging from 2.8x to 7.9x 
larger than the 95% VaR estimate. For Tesla, the range was 1.3x to 3.7x 
larger — proportionally smaller because Tesla's already-elevated VaR 
reflects its high baseline volatility.

**For highly volatile stocks, VaR's proportional underestimation of 
crisis risk is smaller.** Tesla's COVID crash scenario ($1,200) represents 
only 1.96x its VaR estimate ($611), while Apple's COVID scenario ($1,200) 
represents 4.18x its VaR estimate ($287). A higher baseline VaR naturally 
absorbs more of the gap between normal bad days and crisis scenarios — 
though both stocks still show VaR falling far short of true tail risk.

## Results

### Three-Method VaR Comparison

| Metric | AAPL | TSLA |
|---|---|---|
| Daily Volatility (Std Dev) | 1.80% | 3.89% |
| Historical VaR (95%) | $287.39 | $611.51 |
| Monte Carlo VaR (95%) | $291.54 | $635.79 |
| Parametric VaR (95%) | $289.74 | $631.89 |
| Historical vs. Monte Carlo Gap | $4.15 | $24.28 |

### In-Sample Backtest Results

| Metric | AAPL | TSLA |
|---|---|---|
| Total Trading Days | 1,002 | 1,002 |
| VaR Breaches | 51 | 51 |
| Expected Breach Rate | 5.0% | 5.0% |
| Actual Breach Rate | 5.09% | 5.09% |
| Calibration Result | Well-calibrated | Well-calibrated |

### Out-of-Sample Backtest Results

| Metric | AAPL | TSLA |
|---|---|---|
| Training Period | 2022–2024 | 2022–2024 |
| Testing Period | 2024–2026 | 2024–2026 |
| VaR Threshold (from training) | -2.91% | -6.36% |
| Total Testing Days | 396 | 396 |
| VaR Breaches (testing period) | 19 | 14 |
| Expected Breach Rate | 5.0% | 5.0% |
| Actual Breach Rate | 4.80% | 3.54% |
| Calibration Result | Well-calibrated | Overestimates risk |

## Key Insights

**1. Volatility drives the gap between methods.**
Tesla's volatility (3.89%) is roughly double Apple's (1.80%), but the 
gap between Historical and Monte Carlo VaR estimates is nearly six times 
larger for Tesla ($24.28 vs. $4.15). This suggests the normal 
distribution assumption underlying both Monte Carlo and Parametric VaR 
breaks down more severely for higher-volatility stocks, whose actual 
return distributions likely exhibit fatter tails than a standard normal 
distribution predicts — a well-documented phenomenon in equity markets 
known as leptokurtosis.

**2. Monte Carlo and Parametric VaR converge as expected.**
Monte Carlo ($291.54 AAPL, $635.79 TSLA) and Parametric ($289.74 AAPL, 
$631.89 TSLA) estimates came within $2 of each other for both stocks. 
Since both methods share the same normal distribution assumption and 
input parameters, this convergence confirms the Monte Carlo simulation 
was implemented correctly — two independent methods agreeing on nearly 
identical answers is a meaningful cross-validation signal.

**3. Out-of-sample testing reveals model stability differences.**
Apple's model remained well-calibrated out-of-sample (4.80% actual vs. 
5.0% expected breach rate), suggesting Apple's risk profile was 
relatively stable across both time periods. Tesla's model overestimated 
risk out-of-sample (3.54% actual vs. 5.0% expected), because its 
training period (2022–2024) included unusually turbulent market 
conditions that did not persist into the testing period (2024–2026). 
This finding illustrates a fundamental challenge in financial risk 
modeling: highly volatile assets are harder to model consistently across 
different market regimes, which is exactly why real risk management teams 
continuously recalibrate their VaR models rather than treating any single 
threshold as permanently valid.

**4. In-sample backtesting is necessary but not sufficient.**
Both stocks showed nearly identical in-sample breach rates (5.09%), 
which looked like equally strong calibration. The out-of-sample test 
revealed meaningful differences hidden by this surface-level agreement — 
demonstrating why out-of-sample validation is the more rigorous and 
informative standard for assessing model reliability.

## Limitations
- VaR estimates the threshold of loss but does not predict the magnitude 
  of losses beyond that threshold. 
- Monte Carlo and Parametric VaR both assume normally distributed 
  returns, which may underestimate tail risk — particularly for volatile 
  assets like Tesla
- The out-of-sample split (2022–2024 training, 2024–2026 testing) means 
  the model was trained partly on a high-volatility period for both 
  stocks, which influenced the threshold estimates
- This model treats each stock independently and does not account for 
  correlation effects in a multi-asset portfolio

## Future Extensions
- Add Conditional VaR (CVaR/Expected Shortfall) to measure severity of 
  losses beyond the VaR threshold
- Extend to a multi-asset portfolio using a covariance matrix to model 
  diversification effects across correlated assets
- Implement rolling-window volatility to capture time-varying risk rather 
  than assuming constant volatility across the full sample period

## Technologies Used
- Python 3.14
- yfinance — real market data
- NumPy — numerical computation and simulation
- pandas — data manipulation
- matplotlib — data visualization
- scipy — statistical distribution functions

## Visualization
![VaR Comparison Chart](var_comparison_aapl_tsla.png)
