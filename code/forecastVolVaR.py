'''
forecastVolVaR_v6.py
Alejandro Cermeño 

This code forecast one day ahead VaR at 99%, 95% confidence through a suit of 
ARCH-type models. Additionally, three backtest tests are applied. time, numpy,
pandas, datetime, itertools, and arch are required.

See README.txt for additional information.

WARNING: It took about 30 hours per time series to run the code on an 8 GB 
         Memory / 4 Intel vCPUs / 160 GB Disk.
'''

import pandas as pd
from time import time
from numpy import (log, sqrt)
from datetime import timedelta
from itertools import product 

start_code = time() # start stopwatch  


# Functions 
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

path = "https://git.io/JX83R"

prices  = pd.read_excel(path, index_col = 0)
returns = 100 * (log(prices) - log(prices.shift(1)))

# Mean specifications

mean_ops       = ['Zero', 'Constant', 'AR'] 

# Variance models

arch_params    = {'vol':'ARCH'}                               # ARCH
garch_params   = {'p':1, 'q':1, 'vol':'GARCH'}                # GARCH  
grj_params     = {'p':1, 'o':1, 'q':1, 'vol':'GARCH'}         # GRJ
#tarch_params  = {'p':1, 'o':1, 'q':1, 'power':1.0}           # TARCH  (not included)
#egarch_params = {'p':1, 'q':1, 'o':1, 'vol': 'EGARCH'}       # EGARCH (not included)
aparch_params  = {'p':1, 'o':1, 'q':1, 'vol':'APARCH'}        # APARCH
figarch_params = {'p':1, 'q':1, 'power':2.0, 'vol':'FIGARCH'} # FIGARCH

variance_ops  = [arch_params, garch_params, grj_params, aparch_params,
                 figarch_params]

# Distributions

dist_ops     = ['normal', 't', 'skewt', 'ged']

h = 1                            # forecasts horizons
ts = returns.iloc[:, 0].dropna() # serie to use
win_size = 250                   # window size
n_preds  = len(ts) - win_size - 1  

##################################
# Volatility and VaR forecasting #
##################################

forecastVolVaR = pd.DataFrame()

for mean, variance, dist in product(mean_ops, variance_ops, dist_ops):

  # Rolling window
  for i in range(0, n_preds, h): # syntax: range(start, stop, step)

    window = ts[0 + i : win_size  + i]

    # model specification and estimation
    mdl = arch_model_v2(window, mean = mean, **variance, dist = dist, 
                     rescale = False).fit(disp = 'off')

    # forecasting
    pred = mdl.forecast(horizon = h, reindex = True)

    # result values
    mean_true = ts[win_size  + i + 1]              # realized mean 
    mean_pred = pred.mean.dropna().iloc[0, 0]      # forecasted men (when AR(1))
    var_pred  = pred.variance.dropna().iloc[0, 0]  # forecasted variance (sigma^2)
    vol_pred  = sqrt(var_pred)                     # forecasted vol. (sigma)
    vol_true  = abs(ts[win_size  + i + 1])         # realized vol. (|r|)
    cond_vol  = mdl.conditional_volatility         # conditional vol.

    # Value at Risk forecast
    try:
      mdl.params['mu']
    except KeyError:
      mdl.params['mu'] = 0

    std_rets = ( (window - mdl.params['mu']) / cond_vol ).dropna()
    q        = std_rets.quantile([0.01, 0.05])
    VaR      = mean_pred + vol_pred * q.values[None, :]

    # Value at Risk forecast
    dist_params_ops = ['nu', 'eta', 'lambda']
    dist_params = list()

    for param in dist_params_ops:
      try:
        dist_params.append(fit.params[param])
      except KeyError:
        pass

    # Quantil, Syntax: Distribution.ppf(pits, parameters=None)
    q = mdl.distribution.ppf([0.01, 0.05], dist_params)
    VaR = - mean_pred - vol_pred * q[None, :]

    # Results table
    to_forecastVolVaR = pd.DataFrame({"serie": ts.name,
                                      "mean": mean,
                                      "variance": mdl.model.volatility.name,
                                      "dist": dist,
                                      "h": h,
                                      "date": ts.index[win_size  + i + 1],
                                      "mean_true": mean_true,
                                      "mean_pred": mean_true,
                                      "var_pred": var_pred,
                                      "vol_true": vol_true,
                                      "vol_pred": vol_pred,
                                      "VaR_1": VaR.ravel()[0],
                                      "VaR_5": VaR.ravel()[1],
                                      "BIC": mdl.bic,
                                      "loglik": [mdl.loglikelihood]
                                      })                  
    
    forecastVolVaR = forecastVolVaR.append(to_forecastVolVaR, sort = False)

# Export table
export(forecastVolVaR, ts.name + '_forecastVolVaR_' + str(n_preds) + '_OOS',
       excel = True)

print('Excel ' + ts.name + '_forecastVolVaR_' + str(n_preds) + '_OOS.xlsx \
saved')

end_code = time() # end stopwatch 
time_code = str(timedelta(seconds = round(end_code - 
                                          start_code))) # Execution time
print('Execution completed')
print('Execution time: ', time_code)
