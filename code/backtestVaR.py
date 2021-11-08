'''
backtestVaR.py
Alejandro Cermeño (09/2021)

The code applies the backtest procedures of Kupiec (1995), Christoffesen (1998)
and Engle and Manganelli (2004) for VaR at 99%, 95% confidence level. The MAE 
and MSE are also calculated for the volatility forecasts. time, numpy, pandas, 
scipy, datetime, itertools and sklearn.metrics are required.

See README.txt for additional information.
'''

import numpy as np
import pandas as pd 
from time import time
from datetime import timedelta
from scipy import stats
from itertools import product 
from sklearn.metrics import mean_absolute_error, mean_squared_error

start_code = time() # start stopwatch  

class varbacktest:
    r"""
    Backtesting procedures for the Value at Risk (VaR)

    The following backtesting procedures can be specified using varbacktest:
        * Traffic light (TL) test (*FORTCOMING*)
        * Unconditional coverage (UC) test 
        * Conditional coverage independence (CCI) test
        * Conditional Coverage (CC) test
        * Dynamic Quantile (DQ) test

    Parameters
    ----------
    returns : {ndarray, Series}
        Description
    VaR : {ndarray, Series}
        Description
    alpha : float, optional
        Description
    hit_lags : int, optional
        Description
    forecast_lags : int, optional
        Description
    """

    def __init__(
        self, returns, VaR, alpha = 0.05, hit_lags = 4, forecast_lags = 1):
      
        self.index = returns.index
        self.returns = returns.values
        self.VaR = VaR.values
        self.alpha = alpha
        self.hit_lags = hit_lags
        self.forecast_lags = forecast_lags

        if len(returns) != len(VaR):
          raise ValueError("Returns and VaR series must have the same lengths")
        #if not isinstance(hit_lags, int) or hit_lags >= 1:
        #  raise ValueError("hit_lags must be a positive integer")
        #if not isinstance(forecast_lags, int) or forecast_lags >= 1:
        #  raise ValueError("forecast_lags must be a positive integer")

    def serie_hits(self):
      return (self.returns < self.VaR) * 1


    def num_hits(self):
      return self.serie_hits().sum()


    def pct_hits(self):
      return self.serie_hits().mean()


    def uc(self):
      """Unconditional coverage test (UC) of Kupiec (1995) also know as 
      Proportion of failures test (POF)"""

      N = len(self.returns) # Number of observation
      x = self.num_hits()   # Number of failures

      if x == 0:
        LRuc = -2 * N * np.log(1 - self.alpha)
      elif x < N:
        LRuc = -2 * ((N - x) * np.log(N * (1 - self.alpha) / (N - x)) + x * 
                     np.log(N * self.alpha / x))
      elif x == N:
        LRuc = -2 * N * np.log(self.alpha)

      dof = 1
      PVuc = (1 - stats.chi2.cdf(LRuc, dof))

      return pd.Series([LRuc, PVuc], index=["LRuc", "PVuc"], name = "UC")


    def cci(self):
      """Conditional coverage independence test (CCI) of Christoffersen (1998)"""
    
      hits = self.serie_hits()   # Hit series
      tr = hits[1:] - hits[:-1]  # Sequence to find transitions

      # Number of periods with no failures followed by a period with failures
      n01 = (tr == 1).sum()
      # Number of periods with failures followed by a period with no failures
      n10 = (tr == -1).sum()
      # Number of periods with failures followed by a period with failures
      n11 = (hits[1:][tr == 0] == 1).sum()
      # Number of periods with no failures followed by a period with no failures
      n00 = (hits[1:][tr == 0] == 0).sum()

      # Times in the states
      n0  = n01 + n00 
      n1 = n10 + n11
      n = len(self.returns)

      # Probabilities of the transitions from one state to another
      p01 = n01 / (n00 + n01)
      p11 = n11 / (n11 + n10)
      p = n1 / n

      if n1 > 0:
        cci_h0 = (n00 + n01) * np.log(1 - p) + (n01 + n11) * np.log(p)
        cci_h1 = n00 * np.log(1 - p01) + n01 * np.log(p01) + n10 * np.log(1 -
                                                                          p11)

        if p11 > 0:
          cci_h1 += n11 * np.log(p11)
        
        LRcci = -2 * (cci_h0 - cci_h1)
        PVcci = (1 - stats.chi2.cdf(LRcci, 1))
      else:
        PVcci = np.nan

      return pd.Series([LRcci, PVcci], index = ["LRcci", "PVcci"], name = "CCI")


    def cc(self):
      """Conditional coverage test (CC) of Christoffersen (1998)"""
    
      LRuc  = self.uc()["LRuc"]   # Unconditional coverage
      LRcci = self.cci()["LRcci"] # Independence
      LRcc = LRuc + LRcci         # Conditional coverage

      PVcc = (1 - stats.chi2.cdf(LRcc, 2))

      return pd.Series([LRcc, PVcc], index=["LRcc", "PVcc"], name = "CC")


    def dq(self):
      """Dynamic quantile test (DQ) of Engle and Manganelli (2004)"""

      try:
        hits = self.serie_hits()
        p, q, n = self.hit_lags, self.forecast_lags, hits.size
        pq = max(p, q - 1)
        y = hits[pq:] - self.alpha  # Dependent variable
        x = np.zeros((n - pq, 1 + p + q))
        x[:, 0] = 1  # Constant

        for i in range(p): # Lagged hits 
          x[:, 1 + i] = hits[pq - (i + 1) : - (i + 1)]

        for j in range(q): # Actual + lagged VaR forecast
          if j > 0:
            x[:, 1 + p + j] = self.VaR[pq - j : - j]
          else:
            x[:, 1 + p + j] = self.VaR[pq:]

        beta = np.dot(np.linalg.inv(np.dot(x.T, x)), np.dot(x.T, y))
        LRdq = np.dot(beta, np.dot(np.dot(x.T, x), beta)) / (self.alpha * 
                                                             (1 - self.alpha))
        PVdq = 1 - stats.chi2.cdf(LRdq, 1 + p + q)

      except:
        LRdq, PVdq = np.nan, np.nan

      return pd.Series([LRdq, PVdq], index=["LRdq", "PVdq"], name = "DQ")


    def summary(self):
      """Run all implemented VaR backtests"""
      df = pd.DataFrame({"VaR_lvl":  self.alpha,
                         "Obs":      len(self.returns),                    
                         "num_hits": self.num_hits(),
                         "pct_hits": [self.pct_hits()],
                         "LRuc":     self.uc()["LRuc"],
                         "PVuc":     self.uc()["PVuc"],
                         "LRcci":    self.cci()["LRcci"],
                         "PVcci":    self.cci()["PVcci"],
                         "LRcc":     self.cc()["LRcc"],
                         "PVcc":     self.cc()["PVcc"],
                         "LRdq":     self.dq()["LRdq"],
                         "PVdq":     self.dq()["PVdq"]
                         })
      return df


    # (*FORTCOMING* traffic light test)
    #def tl(self)      
      #light = ["green", "yellow", "red"]
      #hits = self.serie_hits()
      #N = len(hits)
      #x = sum(hits)
      #
      #Probability = stats.binom.cdf(x, N, pVaR)


# Function

def export(df, file_name, excel = None, latex = None):

  # To Excel
  if excel == True:
    df.to_excel(file_name + '.xlsx')  
        
  # To LaTeX
  if latex == True:
    latex_code = df.to_latex()
    with open(file_name + '.tex', 'w') as tex:
      tex.write(latex_code)


######################################
# Data collection and specifications #
######################################

path = "/content/SPBLPGPT_forecastVolVaR_4853_OOS.xlsx"
df = pd.read_excel(path, index_col = 0)

df['VaR_1'] = df['VaR_1'].apply(lambda x: x*-1)
df['VaR_5'] = df['VaR_5'].apply(lambda x: x*-1)

# specifications

mean_ops = df['mean'].unique()
variance_ops = df['variance'].unique()
dist_ops = df['dist'].unique()
VaR_ops = ['VaR_1', 'VaR_5']
conf_lvl_ops = [0.01, 0.05]

backtestVaR = pd.DataFrame() 

for mean, variance, dist in product(mean_ops, variance_ops, dist_ops):

  filtered = df[(df['mean'] == mean) &
                (df['variance'] == variance) &
                (df['dist'] == dist)]

  ##############################
  # Backtest and other metrics #
  ##############################

  returns = filtered['mean_true']
  vol_true = filtered['vol_true'].values
  vol_pred = filtered['vol_pred'].values

  mse = mean_squared_error(vol_true, vol_pred)  # MSE 
  mae = mean_absolute_error(vol_true, vol_pred) # MAE

  # For each confidence level 
  for i in range(len(VaR_ops)):

    bt = varbacktest(returns,
                     VaR = filtered[VaR_ops[i]], # Select column 'VaR_1' or 'VaR_5' 
                     alpha = conf_lvl_ops[i]
                     )

    # Results table
    add = pd.concat([pd.DataFrame({"serie": filtered['serie'].unique(),
                                      "mean": mean,
                                      "variance": variance,
                                      "dist": dist,
                                      "mae": mae,
                                      "mse": mse}),
                     bt.summary()],
                     axis = 1)      
        
    backtestVaR = backtestVaR.append(add)

# The results are exported
export(backtestVaR, 'backtestVaR_' + df['serie'].unique()[0], excel = True)

end_code = time() # end stopwatch 
time_code = str(timedelta(seconds = round(end_code - 
                                          start_code))) # Execution time

print('backtestVaR_' + df['serie'].unique()[0] + '.xlsx successfully saved')
print('Execution completed')
