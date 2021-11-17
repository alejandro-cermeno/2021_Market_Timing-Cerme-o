import numpy as np
import pandas as pd 
from scipy import stats

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
        Contains the returns values.
    VaR : {ndarray, Series}
        Contains Value-at-Risk (VaR) values. Should be in the same units as the returns data. 
    alpha : float, optional
        Contains the desired VaR confidence level. Default value is 0.05.
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

      # Probability of having a failure on period t, given that  there was no failure on period t-1
      p01 = n01 / (n00 + n01)
      # Probability of having a failure on period t, given that there was a failure on period t-1
      p11 = n11 / (n11 + n10)
      # Probability of having a failure on period t
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

      dof = 2
      PVcc = 1 - stats.chi2.cdf(LRcc, dof)

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
        DQ = np.dot(beta, np.dot(np.dot(x.T, x), beta)) / (self.alpha * 
                                                             (1 - self.alpha))
        PVdq = 1 - stats.chi2.cdf(DQ, 1 + p + q)

      except:
        DQ, PVdq = np.nan, np.nan

      return pd.Series([DQ, PVdq], index=["DQ", "PVdq"], name = "DQ")


    def summary(self):
      """Run all implemented VaR backtests"""
      df = pd.DataFrame({"VaR_lvl":  self.alpha,
                         "obs":      len(self.returns),                    
                         "num_hits": self.num_hits(),
                         "pct_hits": [self.pct_hits()],
                         "LRuc":     self.uc()["LRuc"],
                         "PVuc":     self.uc()["PVuc"],
                         "LRcci":    self.cci()["LRcci"],
                         "PVcci":    self.cci()["PVcci"],
                         "LRcc":     self.cc()["LRcc"],
                         "PVcc":     self.cc()["PVcc"],
                         "DQ":     self.dq()["DQ"],
                         "PVdq":     self.dq()["PVdq"]
                         })
      return df

# Data collection
path = "https://git.io/J1clf"
df = pd.read_excel(path, index_col = 0)

# specifications
VaR_ops = ['VaR_1', 'VaR_5']
conf_lvl_ops = [0.01, 0.05]
returns = df['mean_true']

# backtest
for i in range(len(VaR_ops)):
  VaR = df[VaR_ops[i]]
  bt = varbacktest(returns, VaR, alpha = conf_lvl_ops[i])
  display(bt.summary())

# VaR_lvl  obs	num_hits      pct_hits	LRuc	        PVuc	  LRcci	        PVcci  	  LRcc          PVcc	  DQ	        PVdq
# 0.01	   1703	166	      0.097475	471.596309	0.0	  294.685777	0.0	  766.282086	0.0	  7517.921735	0.0

# VaR_lvl	obs	  num_hits	pct_hits	LRuc	      PVuc	LRcci	      PVcci	LRcc	      PVcc	DQ	        PVdq
# 0.05	        1703	  284	        0.166765	311.998462    0.0	198.32140     0.0	510.319863    0.0	1659.205151	0.0