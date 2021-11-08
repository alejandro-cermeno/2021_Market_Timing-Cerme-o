# ARCH models for Value at Risk forecasting in Latin American stock and Forex markets

#### This is the the supplementary material that allows to replicate the results tables, and access to the source data and full results.

### Abstract

Using stock and Forex markets daily returns, a set of ARCH models with three different
mean specifications and four error distributions are used to forecast the one-day ahead Value
at Risk (VaR) at 1% and 5% confidence level for a group of Latin American countrys. The
main results are: (i) in general, FIGARCH volatility process and leptokurtic distributions
are able to produce better one-step-ahead VaR forecasts (ii) the models that best fit the full
series in-sample are not necessarily the ones that obtain the most accurate VaR forecasts
out-of-sample and (iii) the models producing the most accurate forecasts vary by market and
country.

### Module Contents

* code: codes to replicate results in draft.pdf
  * data.xlsx: source data used as obtained from Bloomberg.
  * descriptiveStats.py: descriptive statistics for stock and Forex markets returns. Replicates table 1.
  * modelingVol_params.py: procedure to estimated parameters. Replicates table 2.
  * forecastVolVaR.py: forecast volatility with a rolling window.
  * backtestVaR.py: procedure to backtest forecast VaR. Replicates table 3.

* output: full obtained outputs
* draft.pdf
* slides.pdf
