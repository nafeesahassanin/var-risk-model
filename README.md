# Value at Risk (VaR) Model — AAPL & TSLA

## Overview
This project builds a Value at Risk (VaR) model in Python to estimate 
the daily risk of a $10,000 investment in Apple stock (AAPL) using 
four years of real market data (2022–2026). Two methods are compared: 
Historical Simulation and Monte Carlo Simulation.

## What is Value at Risk?
Value at Risk answers the question: "How much could this investment 
lose on a bad day, and how confident are we in that estimate?"

A 95% confidence VaR of $287 means: on 95% of trading days, losses 
will not exceed $287. On the worst 5% of days — roughly 12–13 days 
per year — losses could exceed that threshold.

## Methodology

### Historical VaR
Uses 1,002 days of real Apple returns (2022–2026) to find the actual 
5th percentile of historical daily losses. No distributional assumptions 
are made — the model uses what actually happened.

### Monte Carlo VaR
Generates 10,000 simulated daily returns drawn from a normal distribution 
parameterized by Apple's real mean return (0.06%) and standard deviation 
(1.80%). Finds the 5th percentile of the simulated distribution.

## Results

| Metric | AAPL | TSLA |
|---|---|---|
| Daily Volatility (Std Dev) | 1.80% | 3.89% |
| Historical VaR (95%) | $287.39 | $611.51 |
| Monte Carlo VaR (95%) | $291.54 | $635.79 |
| Historical vs MC Gap | $4.15 | $24.28 |

## Key Insight
Comparing Apple and Tesla reveals that the gap between Historical and 
Monte Carlo VaR estimates scales with volatility — Tesla's gap is roughly 
6x larger than Apple's, despite its volatility being only about 2x higher. 
This suggests the normal distribution assumption underlying Monte Carlo 
simulation breaks down more severely for higher-volatility stocks, whose 
actual return distributions likely exhibit fatter tails than a standard 
normal distribution predicts.


## Limitations
- VaR does not predict the magnitude of losses beyond the threshold
- The Monte Carlo model assumes normally distributed returns, which 
  may underestimate tail risk in more volatile stocks or crisis periods
- Historical VaR is limited by the specific market conditions in the 
  sample period

## Technologies Used
- Python 3.14
- yfinance — real market data
- NumPy — numerical computation and simulation
- pandas — data manipulation
- matplotlib — visualization
- scipy — statistical functions

## Visualization
![VaR Analysis Chart](var_analysis.png)
