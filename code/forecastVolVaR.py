'''
forecastVolVaR_v5.py
Alejandro Cermeño (09/2021)

This code makes projections one period forward of the volatility of 
each series entered through ARCH models. These consider three specifications 
of the mean (zero, constant and autoregressive), five of the variance (ARCH, 
GARCH, GJR, APARCH and FIGARCH), and four distributions (normal, t-student, 
skeewed student-t and GED). Likewise, with the volatility projections the 
Value at Risk is projected at 99%, 95% confidence. 
The time, numpy, pandas, datetime, itertools, and arch libraries are required.

See README.txt for additional information.
'''

import time
import numpy as np
import pandas as pd

from datetime import timedelta
from itertools import product 
from arch import arch_model
from arch.__future__ import reindexing


start_code = time.time() # stopwatch start 


#############
# Functions #
#############

def price2ret(price):
  ret = (100 * (np.log(price) - np.log(price.shift(1))))
  return ret


def export(df, file_name, Excel = None, LaTeX = None):

  # Exportar a Excel

  if Excel == True:

    df.to_excel(file_name + '.xlsx')  

  # Exportar a LaTeX

  if LaTeX == True:
    latex_code = df.to_latex()

    with open(file_name + '.tex', 'w') as tex:
      tex.write(latex_code)


def name2disp(especificacion):

  ''' 
  Para cada especificación de la media, varianza o distribucion, establece
  el nombre de la especificación a mostrarse.
  '''

  name2disp_ = ''
  
  # Nomnbre especificacion de la media
  
  if especificacion == 'Zero':
    name2disp_ = 'Cero'
  elif especificacion == 'Constant':
    name2disp_ = 'Constante'
  elif especificacion == 'AR':
    name2disp_ = 'AR'

  # Nombre modelo de la varianza

  elif especificacion == arch_params:
    name2disp_ = 'ARCH'
  elif especificacion == garch_params:
    name2disp_ = 'GARCH'
  elif especificacion == grj_params:
    name2disp_ = 'GJR'
  elif especificacion == egarch_params:
    name2disp_ = 'EGARCH'
  elif especificacion == aparch_params:
    name2disp_ = 'APARCH'
  elif especificacion == figarch_params:
    name2disp_ = 'FIGARCH'

  # Nombre de la distribucion

  elif especificacion == 'normal':
    name2disp_ = 'N'
  elif especificacion == 't':
    name2disp_ = 't'
  elif especificacion == 'skewt':
    name2disp_ = 'skt'
  elif especificacion == 'ged':
    name2disp_ = 'GED'

  else:
    name2disp_ = 'Especificacion no valida'

  return name2disp_


def signif_ast(coef, pvalues):


def model_info_fun(ts, mean, variance, dist):
    
  # Serie
  serie_name = pd.Series(ts.name, index = ['serie'])
  # Mean
  media_name = pd.Series(name2disp(mean), index = ['mean'])
  # Variance
  varianza_name = pd.Series(name2disp(variance), index=['variance'])
  # Distribution
  dist_name = pd.Series(name2disp(dist), index = ['dist'])

  info = pd.concat([serie_name, media_name, varianza_name, dist_name])

  return info


###################
# Data collection #
###################

path = 'https://git.io/JuGLW'

# Stock market data
stocks  = pd.read_excel(path, sheet_name = 'stocks_raw', index_col = 0)
r_stocks = price2ret(stocks)

# Forex market data  
forex  = pd.read_excel(path, sheet_name = 'forex_raw', index_col = 0)
r_forex = 100 * (np.log(forex) - np.log(forex.shift(1)))

returns = r_forex.join(r_stocks)


##################
# Specifications #
##################

# Mean specifications

mean_ops       = ['Zero', 'Constant', 'AR'] 

# Variance models

arch_params    = {'vol':'ARCH'}                               # ARCH
garch_params   = {'p':1, 'q':1, 'vol':'GARCH'}                # GARCH  
grj_params     = {'p':1, 'o':1, 'q':1, 'vol':'GARCH'}         # GRJ
#tarch_params   = {'p':1, 'o':1, 'q':1, 'power':1.0}          # TARCH  (not included)
#egarch_params  = {'p':1, 'q':1, 'o':1, 'vol': 'EGARCH'}      # EGARCH (not included)
aparch_params  = {'p':1, 'o':1, 'q':1, 'vol':'APARCH'}        # APARCH
figarch_params = {'p':1, 'q':1, 'power':2.0, 'vol':'FIGARCH'} # FIGARCH

variance_ops  = [arch_params, garch_params, grj_params, aparch_params,
                 figarch_params]

# Distributions

dist_ops     = ['normal', 't', 'skewt', 'ged']

h = 1 #, 5, 10] # forecasts horizons

# parameters

ts = returns.iloc[:, 0].dropna() # serie to use
win_size = 250
n_preds  = len(ts) - win_size - 1

################################
# Proyeccion volatilidad y VaR #
################################

# The results are saved in the variable 'serie'_forecastVolVaR_'n_preds'_OOS
# MERVAL_forecastVolVaR_1250_OOS, for example

globals()[ts.name + '_forecastVolVaR_' + str(n_preds) + '_OOS'] = pd.DataFrame()

for mean, variance, dist in product(mean_ops, variance_ops, dist_ops):

  # Rolling window
  for i in range(0, n_preds, h): # syntax: range(start, stop, step)

    window = ts[0 + i : win_size  + i]

    start_model = time.time() # stopwatch for each model


    # model specification
    mdl = arch_model(window, mean = mean, **variance, dist = dist, 
                     rescale = False)

    # model estimation
    fit = mdl.fit(disp = 'off')

    # forecast
    pred = fit.forecast(horizon = h, reindex = True)


    # Valores resultados

    mean_true = ts[win_size  + i + 1]              # realized mean 
    mean_pred = pred.mean.dropna().iloc[0, 0]      # forecasted men (when AR(1))
    var_pred  = pred.variance.dropna().iloc[0, 0]  # forecasted variance (sigma^2)
    vol_pred  = np.sqrt(var_pred)                  # forecasted vol. (sigma)
    vol_true  = abs(ts[win_size  + i + 1])         # realized vol. (|r|)

    # Value at Risk forecast #
    # parametric VaR

    dist_params_ops = ['nu', 'eta', 'lambda']

    dist_params = list()

    for param in dist_params_ops:
      try:
        dist_params.append(fit.params[param])

      except KeyError:
        pass

    # Cuantil. Syntax: Distribution.ppf(pits, parameters=None)
    q = mdl.distribution.ppf([0.01, 0.05], dist_params)

    # Value at Risk
    VaR = - mean_pred - vol_pred * q[None, :]


    # Results table for a specific model
    
    # Columns: series, mean, variance, distribution, forecast horizon (h), date,
    # forecasted mean, realized volatility (|r|), volatility forecasted, 
    # VaR [1%, 5%, 10%], BIC, loglik, coefficients

    mdl_info2t = model_info_fun(ts, mean, variance, dist) # mean, var. and dist.

    h2t = pd.Series(h, index = ['h']) 
    date2t = pd.Series(ts.index[win_size  + i + 1], index = ['date']) 
    mean_true2t = pd.Series(mean_true, index = ['mean_true']) 
    mean_pred2t = pd.Series(mean_pred, index = ['mean_pred']) 
    var_pred2t = pd.Series(var_pred, index = ['var_pred']) 
    vol_true2t = pd.Series(vol_true, index = ['vol_true']) 
    vol_pred2t = pd.Series(vol_pred, index = ['vol_pred'])
    VaR2t = pd.Series(VaR.ravel(), index = ['VaR_1', 'VaR_5']) 


    end_model = time.time()                   # end stopwatch each model
    time_model = str(end_model - start_model) # excecution time  
    time_model2t = pd.Series(time_model, index = ['time'])


    to_forecastVolVaR = pd.concat([mdl_info2t, h2t, date2t, mean_true2t,
                                   mean_pred2t, vol_true2t, vol_pred2t,
                                   VaR2t, time_model2t])
    
    to_forecastVolVaR = pd.DataFrame(to_forecastVolVaR).T

    # Is attached to the results table
    globals()[ts.name + '_forecastVolVaR_' + str(n_preds) + '_OOS'] = globals()[ts.name + '_forecastVolVaR_' + str(n_preds) + '_OOS'].append(to_forecastVolVaR,
                                                                                                                                             sort = False, 
                                                                                                                                             )
# The results table is exported

export(globals()[ts.name + '_forecastVolVaR_' + str(n_preds) + '_OOS'], ts.name + '_forecastVolVaR_' + str(n_preds) + '_OOS', Excel = True)

print('El Excel ' + ts.name + '_forecastVolVaR_' + str(n_preds) + '_OOS.xlsx \
ha sido guardado con exito')


end_code = time.time() # End stopwatch
time_code = str(timedelta(seconds = round(end_code - start_code))) # Excecution time

print('Execution completed')
print('Execution time: ', time_code)
