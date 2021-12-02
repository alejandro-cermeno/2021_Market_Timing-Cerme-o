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

#####################
# varbacktest class #
#####################

class varbacktest:
    r"""
    Backtesting procedures for the Value at Risk (VaR)
    The following backtesting procedures can be specified using varbacktest:
        * Traffic light (TL) test (*FORTCOMING*)
        * Unconditional coverage (UC) test
        * Conditional coverage independence (CCI) test
        * Conditional coverage (CC) test
        * Dynamic Quantile (DQ) test
    Additionally,
        * Actual over expected ratio
        * Tick loss function
        * Firm’s loss function
        * Quadratic loss function
        
    Parameters
    ----------
    returns : {ndarray, Series}
        Contains the returns values.
    VaR : {ndarray, Series}
        Contains Value-at-Risk (VaR) values. Should be in the same units as the
        returns data.
    alpha : float, optional
        Contains the desired VaR significance level. Default value is 0.05.
    hit_lags : int, optional
        Description
    forecast_lags : int, optional
        Description
    """

    def __init__(self, returns, VaR, alpha=0.05, hit_lags=4, forecast_lags=1):

        self.index = returns.index
        self.returns = returns.values
        self.VaR = VaR.values
        self.alpha = alpha
        self.hit_lags = hit_lags
        self.forecast_lags = forecast_lags

        if len(returns) != len(VaR):
            raise ValueError(
                "Returns and VaR series must have the same lengths"
                )
        # if not isinstance(hit_lags, int) or hit_lags >= 1:
        #  raise ValueError("hit_lags must be a positive integer")
        # if not isinstance(forecast_lags, int) or forecast_lags >= 1:
        #  raise ValueError("forecast_lags must be a positive integer")

    def serie_hits(self):
        return (self.returns < self.VaR) * 1  # <- Ardia et al. (2019) use

    def num_hits(self):
        return self.serie_hits().sum()

    def pct_hits(self):
        return self.serie_hits().mean()

    def ae(self):
        """Actual over expected ratio"""
        N = len(self.returns)  # Number of observation
        x = self.num_hits()  # Number of failures
        return x / (self.alpha * N)  # Actual / Expected

    def tick_loss(self, return_mean=True):
        """Tick loss function of González-Rivera et al. (2004)"""
        loss = (self.alpha - self.serie_hits()) * (self.returns - self.VaR)
        if return_mean:
            return loss.mean()
        else:
            return loss

    def firm_loss(self, c=1, return_mean=True):
        """Firm’s loss function of Sarma et al. (2003)"""
        loss = (
            self.serie_hits() * (1 + (self.returns - self.VaR) ** 2)
            - c * (1 - self.serie_hits()) * self.VaR
        )
        if return_mean:
            return loss.mean()
        else:
            return loss

    def quadratic_loss(self, return_mean=True):
        """Quadratic loss function of Lopez (1999), and Martens et al. (2009)"""
        loss = self.serie_hits() * (1 + (self.returns - self.VaR) ** 2)
        if return_mean:
            return loss.mean()
        else:
            return loss

    def uc(self):
        """Unconditional coverage test (UC) of Kupiec (1995) also know as
        Proportion of failures test (POF)"""

        N = len(self.returns)  # Number of observation
        x = self.num_hits()  # Number of failures

        if x == 0:
            LRuc = -2 * N * np.log(1 - self.alpha)
        elif x < N:
            LRuc = -2 * (
                (N - x) * np.log(N * (1 - self.alpha) / (N - x))
                + x * np.log(N * self.alpha / x)
            )
        elif x == N:
            LRuc = -2 * N * np.log(self.alpha)

        dof = 1
        PVuc = 1 - stats.chi2.cdf(LRuc, dof)

        if PVuc > self.alpha:
          UC = 'accept'
        else: 
          UC = 'reject'

        return pd.Series([LRuc, PVuc, UC], index=["LRuc", "PVuc", "UC"], 
                         name="UC_test")

    def cci(self):
        """Conditional coverage independence test (CCI) of Christoffersen (1998)"""

        hits = self.serie_hits()  # Hit series
        tr = hits[1:] - hits[:-1]  # Sequence to find transitions

        # Number of periods with no failures followed by a period with failures
        n01 = (tr == 1).sum()
        # Number of periods with failures followed by a period with no failures
        n10 = (tr == -1).sum()
        # Number of periods with failures followed by a period with failures
        n11 = (hits[1:][tr == 0] == 1).sum()
        # Number of periods with no failures followed by a period with no failures
        n00 = (hits[1:][tr == 0] == 0).sum()

        LogLNum = 0
        if (n00 + n10) > 0 and (n01 + n11) > 0:
            pUC = (n01 + n11) / (n00 + n01 + n10 + n11)
            LogLNum = (n00 + n10) * np.log(1 - pUC) + (n01 + n11) * np.log(pUC)

        LogLDen = 0
        if n00 > 0 and n01 > 0:
            p01 = n01 / (n00 + n01)
            LogLDen = LogLDen + n00 * np.log(1 - p01) + n01 * np.log(p01)
        if n10 > 0 and n11 > 0:
            p11 = n11 / (n10 + n11)
            LogLDen = LogLDen + n10 * np.log(1 - p11) + n11 * np.log(p11)

        LRcci = -2 * (LogLNum - LogLDen)
        dof = 1
        PVcci = 1 - stats.chi2.cdf(LRcci, dof)

        if PVcci > self.alpha:
          CCI = 'accept'
        else: 
          CCI = 'reject'

        return pd.Series([LRcci, PVcci, CCI], index=["LRcci", "PVcci", "CCI"], 
                         name="CCI_test")

    def cc(self):
        """Conditional coverage test (CC) of Christoffersen (1998)"""

        LRuc = self.uc()["LRuc"]  # Unconditional coverage
        LRcci = self.cci()["LRcci"]  # Independence
        LRcc = LRuc + LRcci  # Conditional coverage

        dof = 2
        PVcc = 1 - stats.chi2.cdf(LRcc, dof)

        if PVcc > self.alpha:
          CC = 'accept'
        else: 
          CC = 'reject'

        return pd.Series([LRcc, PVcc, CC], index=["LRcc", "PVcc", "CC"], name="CC_test")

    def dq(self):
        """Dynamic quantile test (DQ) of Engle and Manganelli (2004)"""

        try:
            hits = self.serie_hits()
            p, q, n = self.hit_lags, self.forecast_lags, hits.size
            pq = max(p, q - 1)
            y = hits[pq:] - self.alpha  # Dependent variable
            x = np.zeros((n - pq, 1 + p + q))
            x[:, 0] = 1  # Constant

            for i in range(p):  # Lagged hits
                x[:, 1 + i] = hits[pq - (i + 1) : -(i + 1)]

            for j in range(q):  # Actual + lagged VaR forecast
                if j > 0:
                    x[:, 1 + p + j] = self.VaR[pq - j : -j]
                else:
                    x[:, 1 + p + j] = self.VaR[pq:]

            beta = np.dot(np.linalg.inv(np.dot(x.T, x)), np.dot(x.T, y))
            DQtest = np.dot(beta, np.dot(np.dot(x.T, x), beta)) / (
                self.alpha * (1 - self.alpha)
            )
            PVdq = 1 - stats.chi2.cdf(DQtest, 1 + p + q)

        except:
            DQtest, PVdq = np.nan, np.nan

        if PVdq > self.alpha: 
          DQ = 'accept'
        else:
          DQ = 'reject'

        return pd.Series([DQtest, PVdq, DQ], index=["DQtest", "PVdq", "DQ"],
                         name="DQ_test")

    def summary(self):
        """Run all implemented VaR backtests"""
        df = pd.DataFrame(
            {
                "VaR_lvl": self.alpha,
                "obs": len(self.returns),
                "num_hits": self.num_hits(),
                "pct_hits": [self.pct_hits()],
                "actual_expected": self.ae(),
                "LRuc": self.uc()["LRuc"],
                "PVuc": self.uc()["PVuc"],
                "UC": self.uc()["UC"],
                "LRcci": self.cci()["LRcci"],
                "PVcci": (self.cci()["PVcci"]),
                "CCI": (self.cci()["CCI"]),
                "LRcc": (self.cc()["LRcc"]),
                "PVcc": (self.cc()["PVcc"]),
                "CC": (self.cc()["CC"]),
                "DQtest": (self.dq()["DQtest"]),
                "PVdq": (self.dq()["PVdq"]),
                "DQ": (self.dq()["DQ"]),
                "firm_loss": (self.firm_loss()),
                "quadratic_loss": (self.quadratic_loss()),
                "tick_loss": (self.tick_loss()),
            }
        )
        return df


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

#df['VaR_1'] = df['VaR_1'].apply(lambda x: x*-1)
#df['VaR_5'] = df['VaR_5'].apply(lambda x: x*-1)

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
