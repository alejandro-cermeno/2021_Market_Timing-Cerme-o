'''
forecastVolVaR.py

Alejandro Cermeño (09/2021)

Este codigo realiza proyecciones un periodo hacia adelante de la volatilidad 
de cada serie ingresada mediante 60 modelos ARCH. Estos consideran tres 
especificaciones de la media (cero, constante y autoregresiva), cinco de la 
varianza (ARCH, GARCH, GJR, EGARCH y APARCH), y cuatro distribuciones (normal,
t-student, t-student sesgada y GED). Así mismo, con las proyecciones de la 
volatilidad se proyecta el Value at Risk al 99%, 95% y 90% de confianza. Se 
requieren las librerías time, numpy, pandas, datetime, itertools y arch.

Se omiten las tildes. Ver README.txt para informacion adicional.
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

  # Nomnbre especificacion de la media

  if especificacion == 'Zero':
    name2disp = 'Cero'
  elif especificacion == 'Constant':
    name2disp = 'Constante'
  elif especificacion == 'AR':
    name2disp = 'AR'

  # Nombre modelo de la varianza

  elif especificacion == arch_params:
    name2disp = 'ARCH'
  elif especificacion == garch_params:
    name2disp = 'GARCH'
  elif especificacion == grj_params:
    name2disp = 'GJR'
  elif especificacion == egarch_params:
    name2disp = 'EGARCH'
  elif especificacion == aparch_params:
    name2disp = 'APARCH'

  # Nombre de la distribucion

  elif especificacion == 'normal':
    name2disp = 'N'
  elif especificacion == 't':
    name2disp = 't'
  elif especificacion == 'skewt':
    name2disp = 'skt'
  elif especificacion == 'ged':
    name2disp = 'GED'

  else:
    name2disp = 'Especificacion no valida'

  return name2disp


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

mean_ops      = ['Zero', 'Constant', 'AR'] 

# Varianza

arch_params   = {'vol': 'ARCH'}
garch_params  = {'p':1, 'q':1, 'vol':'GARCH'}
grj_params    = {'p':1, 'o':1, 'q':1, 'vol':'GARCH'}
egarch_params = {'p': 1, 'q': 1, 'o': 1, 'vol': 'EGARCH'}
aparch_params = {'p':1, 'o':1, 'q':1, 'power': 2.0, 'vol':'GARCH'}

variance_ops  = [arch_params, garch_params, grj_params, egarch_params, 
                 aparch_params]

# Distribuciones

dist_ops    = ['normal', 't', 'skewt', 'ged']

# Horizontes de proyeccion

h = 1 #, 5, 10]

# Parametros

ts = returns.iloc[:, 0].dropna() # Serie a utilizar
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


    # Especificacion y estimacion del modelo
    mdl = arch_model(window, mean = mean, **variance, dist = dist).fit(disp =
                                                                       'off')

    # Proyeccion
    pred = mdl.forecast(horizon = h, reindex = True)


    # Valores resultados

    mean_true = ts[win_size  + i + 1]              # Media realizada 
    mean_pred = pred.mean.dropna().iloc[0, 0]      # Media proyectada
    var_pred  = pred.variance.dropna().iloc[0, 0]  # Varianza proyectada (sigma^2)
    vol_pred  = np.sqrt(var_pred)                  # Vol. proyectada (sigma)
    vol_true  = abs(ts[win_size  + i + 1])         # Vol. realizada (|r|) 
    cond_vol  = mdl.conditional_volatility         # Vol. condicional


    # Proyeccion del Value at Risk

    try:
      mdl.params['mu']
    except KeyError:
      mdl.params['mu'] = 0

    std_rets = ( (window - mdl.params['mu']) / cond_vol ).dropna()
    q        = std_rets.quantile([0.01, 0.05, 0.1])
    VaR      = - mean_pred - vol_pred * q.values[None, :]


    # Tabla de resultados para un modelo en específico

    # Columnas: serie, media, varianza, distribucion, horizonte de proyeccion
    # (h), fecha, proyeccion media, volatilidad realizada (|r|), volatilidad
    # proyectada, VaR [1%, 5%, 10%], BIC, loglik, coeficientes.

    mdl_info2t = model_info_fun(ts, mean, variance, dist)

    h2t = pd.Series(h, index = ['h'])

    date2t = pd.Series(ts.index[win_size  + i + 1], index = ['date']) 

    mean_true2t = pd.Series(mean_true, index = ['mean_true'])

    mean_pred2t = pd.Series(mean_pred, index = ['mean_pred'])

    var_pred2t = pd.Series(var_pred, index = ['var_pred'])

    vol_true2t = pd.Series(vol_true, index = ['vol_true'])

    vol_pred2t = pd.Series(vol_pred, index = ['vol_pred'])
                      
    VaR2t = pd.Series(VaR.ravel(), index = ['VaR_1', 'VaR_5', 'VaR_10'])

    bic2t = pd.Series(mdl.bic, index = ['BIC'])

    loglik2t = pd.Series(mdl.loglikelihood, index = ['loglik'])

    end_model = time.time() # Fin cronometro cada modelo
    time_model = str(end_model - start_model) # Tiempo ejecucion  
    time_model2t = pd.Series(time_model, index = ['time'])

    #coef = signif_ast(round(mdl.params, 3), mdl.pvalues).T

    to_forecastVolVaR = pd.concat([mdl_info2t, h2t, date2t, mean_true2t,
                                   mean_pred2t, vol_true2t, vol_pred2t,
                                   VaR2t, bic2t, loglik2t, time_model2t])
    
    to_forecastVolVaR = pd.DataFrame(to_forecastVolVaR).T
    #to_forecastVolVaR = to_forecastVolVaR.assign(**coef)

    # Se anexa a la tabla de resultados

    globals()[ts.name + '_forecastVolVaR_' + str(n_preds) + '_OOS'] = globals()[ts.name + '_forecastVolVaR_' + str(n_preds) + '_OOS'].append(to_forecastVolVaR,
                                                                                                                                             sort = False)

# Se exporta la tabla de resultados

export(globals()[ts.name + '_forecastVolVaR_' + str(n_preds) + '_OOS'], ts.name + '_forecastVolVaR_' + str(n_preds) + '_OOS', Excel = True)

print('Excel ' + ts.name + '_forecastVolVaR_' + str(n_preds) + '_OOS.xlsx \
guardado con exito')


end_code = time.time() # Fin cronometro
time_code = str(timedelta(seconds = round(end_code - start_code))) # Tiempo ejecucion

print('Ejecucion finalizada')
print('Tiempo de ejecución: ', time_code)