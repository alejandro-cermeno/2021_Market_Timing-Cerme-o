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


#############
# Librerias #
#############

import time
import numpy as np
import pandas as pd

from datetime import timedelta
from itertools import product 
from arch import arch_model
from arch.__future__ import reindexing


start_code = time.time()  # Inicio cronometro para todo el codigo


#############
# Funciones #
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
 
  signif_ast = list()

  for p in range(len(pvalues)):
    if pvalues[p] <= 0.01:
      asterisco = '***' # significativo al 1%                
    elif pvalues[p] <= 0.05:                
      asterisco = '**'  # significativo al 5%                   
    elif pvalues[p] <= 0.1:
      asterisco = '*'   # significativo al 10%                    
    else:
      asterisco = ''    # no significativo     

    valor = str(coef[p]) + asterisco # Se une el coeficiente y el asterisco

    signif_ast.append(valor)

  signif_ast = pd.DataFrame(signif_ast, index = coef.index)
  
  return signif_ast 


def model_info_fun(ts, mean, variance, dist):
    
  # Serie
  serie_name = pd.Series(ts.name, index = ['serie'])

  # Media
  media_name = pd.Series(name2disp(mean), index = ['mean'])

  # Varianza
  varianza_name = pd.Series(name2disp(variance), index=['variance'])

  # Distribucion
  dist_name = pd.Series(name2disp(dist), index = ['dist'])

  info = pd.concat([serie_name, media_name, varianza_name, dist_name])

  return info


######################
# Obtencion de datos #
######################

path = 'https://git.io/JuGLW'

  # Series del mercado bursatil

stocks  = pd.read_excel(path, sheet_name = 'stocks_raw', index_col = 0)
r_stocks = price2ret(stocks)

  # Series del mercado cambiario
  
forex  = pd.read_excel(path, sheet_name = 'forex_raw', index_col = 0)
r_forex = price2ret(forex)

returns = r_forex.join(r_stocks)


####################
# Especificaciones #
####################

# Media

mean_ops       = ['Zero', 'Constant', 'AR'] 

# Varianza

arch_params    = {'vol': 'ARCH'}
garch_params   = {'p':1, 'q':1, 'vol':'GARCH'}
grj_params     = {'p':1, 'o':1, 'q':1, 'vol':'GARCH'}
egarch_params  = {'p': 1, 'q': 1, 'o': 1, 'vol': 'EGARCH'}
aparch_params  = {'p':1, 'o':1, 'q':1, 'power': 2.0, 'vol':'GARCH'}
figarch_params = {'p':1, 'q':1, 'power': 2.0, 'vol':'FIGARCH'}

variance_ops  = [arch_params, garch_params, grj_params, egarch_params, 
                 aparch_params, figarch_params]

# Distribuciones

dist_ops     = ['normal', 't', 'skewt', 'ged']

# Horizontes de proyeccion

h = 1 #, 5, 10]

# Parametros

ts = returns.iloc[:, 0].dropna()[:400] # Serie a utilizar
win_size = 250
n_preds  = len(ts) - win_size - 1

################################
# Proyeccion volatilidad y VaR #
################################

# Se guardan los resultados en la variable 'serie'_forecastVolVaR_'n_preds'_OOS
# MERVAL_forecastVolVaR_1250_OOS, por ejemplo

globals()[ts.name + '_forecastVolVaR_' + str(n_preds) + '_OOS'] = pd.DataFrame()

for mean, variance, dist in product(mean_ops, variance_ops, dist_ops):

  # Ventana movil
  for i in range(0, n_preds, h): # syntax: range(start, stop, step)

    window = ts[0 + i : win_size  + i]

    start_model = time.time() # Inicio cronometro para cada modelo


    # Especificacion
    mdl = arch_model(window, mean = mean, **variance, dist = dist, 
                     rescale = False)

    # Estimacion
    fit = mdl.fit(disp = 'off')

    # Proyeccion
    pred = fit.forecast(horizon = h, reindex = True)


    # Valores resultados

    mean_true = ts[win_size  + i + 1]              # Media realizada 
    mean_pred = pred.mean.dropna().iloc[0, 0]      # Media proyectada
    var_pred  = pred.variance.dropna().iloc[0, 0]  # Varianza proyectada (sigma^2)
    vol_pred  = np.sqrt(var_pred)                  # Vol. proyectada (sigma)
    vol_true  = abs(ts[win_size  + i + 1])         # Vol. realizada (|r|)

    # Proyeccion del Value at Risk #
    # VaR parametrico

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


    # Tabla de resultados para un modelo en específico

    # Columnas: serie, media, varianza, distribucion, horizonte de proyeccion
    # (h), fecha, proyeccion media, volatilidad realizada (|r|), volatilidad
    # proyectada, VaR [1%, 5%, 10%], BIC, loglik, coeficientes.

    mdl_info2t = model_info_fun(ts, mean, variance, dist) # Media, var. y dist.

    h2t = pd.Series(h, index = ['h']) 

    date2t = pd.Series(ts.index[win_size  + i + 1], index = ['date']) 

    mean_true2t = pd.Series(mean_true, index = ['mean_true']) 

    mean_pred2t = pd.Series(mean_pred, index = ['mean_pred']) 

    var_pred2t = pd.Series(var_pred, index = ['var_pred']) 

    vol_true2t = pd.Series(vol_true, index = ['vol_true']) 

    vol_pred2t = pd.Series(vol_pred, index = ['vol_pred'])
                      
    VaR2t = pd.Series(VaR.ravel(), index = ['VaR_1', 'VaR_5']) 


    end_model = time.time()                   # Fin cronometro cada modelo
    time_model = str(end_model - start_model) # Tiempo ejecucion  
    time_model2t = pd.Series(time_model, index = ['time'])


    to_forecastVolVaR = pd.concat([mdl_info2t, h2t, date2t, mean_true2t,
                                   mean_pred2t, vol_true2t, vol_pred2t,
                                   VaR2t, time_model2t])
    
    to_forecastVolVaR = pd.DataFrame(to_forecastVolVaR).T

    # Se anexa a la tabla de resultados

    globals()[ts.name + '_forecastVolVaR_' + str(n_preds) + '_OOS'] = globals()[ts.name + '_forecastVolVaR_' + str(n_preds) + '_OOS'].append(to_forecastVolVaR,
                                                                                                                                             sort = False, 
                                                                                                                                             )
# Se exporta la tabla de resultados

export(globals()[ts.name + '_forecastVolVaR_' + str(n_preds) + '_OOS'], ts.name + '_forecastVolVaR_' + str(n_preds) + '_OOS', Excel = True)

print('El Excel ' + ts.name + '_forecastVolVaR_' + str(n_preds) + '_OOS.xlsx \
ha sido guardado con exito')


end_code = time.time() # Fin cronometro
time_code = str(timedelta(seconds = round(end_code - start_code))) # Tiempo ejecucion

print('Ejecucion finalizada')
print('Tiempo de ejecución: ', time_code)
