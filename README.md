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
  * forecastVolVaR.py: forecast one day ahead VaR at 99% and 95% confidence through a suit of ARCH-type models.
  * backtestVaR.py: applies a suite of backtest procedures for forecasted VaR. Replicates table 3. Also, the MAE and MSE are calculated for the volatility forecasts. 

* output: full obtained outputs
  * descriptiveStats.lyx: Table 1 Descriptive Statistics for Stock and Forex Markets Returns
  * modelingVol_params.xlsx: full Table 2 and Table 3 Estimated Parameters for daily Latin American Stock and Forex Markets Return
  * backtestVaR.xlsx: full Table 4 and 5 Accuracy of VaR predictions for Stock and Forex Markets Returns
  * forecastVolVaR.rar: full volatility and VaR forecasts for each series. Not included because it is very heavy. Available in https://drive.google.com/file/d/1_ExWrolC-KZ8x5QnKZXrcNGr-uKWj1uK/view?usp=sharing.
  
* varbacktest_test: Verify that the `varbacktest()` class is well implemented by comparing results with those of MATLAB and R.
  * toyserie.xlsx
  * varbacktest_testMATLAB.m
  * varbacktest_testPYTHON.py: source code of the `varbacktest()` class
  * varbacktest_testR.R
* draft.pdf
* slides.pdf
