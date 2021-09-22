# Librerias 

import pandas as pd 

from datetime import timedelta
from itertools import product 
from sklearn.metrics import mean_absolute_error, mean_squared_error

#############
# Funciones #
#############

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


# Obtencion de datos 

df = pd.read_excel('/content/BRL_forecastVolVaR_5553_OOS.xlsx')

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


backtestVaR = pd.DataFrame() # Almacenamiento de resultados

for mean, variance, dist in product(mean_ops, variance_ops, dist_ops):

  # Se filtran los resultados del modelo mean-variance-dist en especifico

  filtered = df[(df['mean'] == name2disp(mean)) &
                (df['variance'] == name2disp(variance)) &
                (df['dist'] == name2disp(dist))]

  ###################################
  # Metricas proyeccion volatilidad #
  ###################################

  returns = filtered['mean_true'].values
  vol_true = filtered['vol_true'].values
  vol_pred = filtered['vol_pred'].values


  # Tiempo de ejecucion
  time_exe = sum(list(map(float, filtered['time'].values)))

  # MSE (vol_true vs vol_pred)
  mse = mean_squared_error(vol_true, vol_pred)

  # MAE (vol_true vs vol_pred)
  mae = mean_absolute_error(vol_true, vol_pred)


  ######################################
  # Metricas proyeccion VaR (backtest) #
  ######################################

  VaR_ops     = ['VaR_1', 'VaR_5']
  conf_lvl_ops = [0.01, 0.05]

  # Para cada nivel de confianza 

  for i in range(len(VaR_ops)):

    VaR = filtered[VaR_ops[i]].values
    conf_lvl = conf_lvl_ops[i]


    # Numero y porcentaje de fallas VaR vs. retornos

    if len(VaR) == len(returns): 
      n_total = len(VaR)

    n_correct = sum(HitSequence(returns, VaR))
    n_fails = n_total - n_correct
    pct_fails = "{:.0%}".format(n_fails / n_total)

    # Pruebas de backtest

    pof = kupiec(returns, VaR, conf_lvl)['Pvalue'] 
    pof = "{:.4f}".format(float(pof))                       # Kupiec (1995)

    cc = christoffersen(returns, VaR, conf_lvl)['Pvalue']
    cc = "{:.4f}".format(float(cc))                         # Christoffersen (1998)

    #dq = engleManganelli(returns, VaR, conf_lvl)['Pvalue'] 
    #dq = "{:.4f}".format(float(dq))                        # Engle and Manganelli (2004)

    #######################
    # Tabla de resultados #
    #######################


    serie = pd.Series(df['serie'], name =df['serie'].iloc[1])
    model_info2t = model_info_fun(serie, mean, variance, dist)

    mae2t = pd.Series(mae, index = ['MAE'])
    mse2t = pd.Series(mse, index = ['MSE'])

    VaR_lvl2t = pd.Series((1 - conf_lvl) * 100, index = ['VaR_lvl'])

    n_fails2t = pd.Series(n_fails, index = ['n_fails'])
    pct_fails2t = pd.Series(pct_fails, index = ['pct_fails'])

    pof2t = pd.Series(pof, index = ['pof'])
    cc2t = pd.Series(cc, index = ['cc'])

    time_exe2t = pd.Series(time_exe, index = ['time_exe'])

    to_backtestVaR = pd.concat([model_info2t, mae2t, mse2t, VaR_lvl2t, n_fails2t,
                                pct_fails2t, pof2t, cc2t, time_exe2t])
    to_backtestVaR = pd.DataFrame(to_backtestVaR).T

    backtestVaR = backtestVaR.append(to_backtestVaR)

# Se exportan los resultados
export(backtestVaR, 'backtestVaR', Excel = True)

print('Archivo backtestVaR.xlsx guardado con éxito')

# Se finaliza el código
print('Ejecución finalizada')