'''
descriptiveStats.py

Alejandro Cermeño (09/2021)

Este codigo genera y exporta las tablas de estadísticos descriptivos para los
retornos (descriptive_stats_ret.tex) y la volatilidad (descriptive_stats_ret.tex)
de las series ingresadas. Requiere los paquetes time, numpy, pandas y datetime.

Se omiten las tildes. Ver README.txt para informacion adicional.
'''

#############
# Librerias #
#############

import time
import numpy as np
import pandas as pd

from datetime import timedelta

start = time.time()  # Inicio cronometro


#############
# Funciones #
#############

def price2ret(price):
  
  ret = (100 * ( np.log(price) - np.log(price.shift(1))))
  
  return ret


def descriptiveStats(df):

  # Se selecciona el numero de observaciones, la desviacion estandar, valor
  # minimo, maximo, simetria y kurtosis.

  descriptive_stats = pd.concat([df.describe()\
                                 .loc[['count', 'mean', 'std', 'min', 'max']].T, 
                                 df.skew().rename('skew'), df.kurt().\
                                 rename('kurt')], axis=1).round(2)

  descriptive_stats['count'] = descriptive_stats['count'].astype(int)

  # Fecha de inicio
  
  fecha_inicio = df.apply(lambda df: df.first_valid_index())
  fecha_inicio = pd.DataFrame(fecha_inicio, columns=['Fecha Inicio'])

  # Fecha de fin
  
  fecha_fin = df.apply(lambda df: df.last_valid_index())
  fecha_fin = pd.DataFrame(fecha_fin, columns=['Fecha Fin'])

  descriptive_stats = pd.concat([fecha_inicio, fecha_fin, 
                                 descriptive_stats], axis = 1)

  return descriptive_stats


def export(df, file_name, excel = None, latex = None):

  # Exportar a Excel

  if excel == True:

    df.to_excel(file_name + '.xlsx')  

  # Exportar a LaTeX

  if latex == True:
    latex_code = df.to_latex()

    with open(file_name + '.tex', 'w') as tex:
      tex.write(latex_code)


######################
# Obtencion de datos #
######################

path = 'https://git.io/JuGLW'

  # Series del mercado bursatil

stocks  = pd.read_excel(path, sheet_name = 'stocks_raw', index_col = 0)

r_stocks = price2ret(stocks)

  # Series del mercado cambiario
  
forex  = pd.read_excel(path, sheet_name = 'forex_raw', index_col = 0)
r_forex = 100 * (np.log(forex) - np.log(forex.shift(1)))

returns = r_forex.join(r_stocks)


########################################
# Estadisticos descriptivos - Retornos #
########################################

# Se genera la tabla 'descriptive_stats_ret'

descriptive_stats_ret = descriptiveStats(returns)

# Se exporta a LaTeX

export(descriptive_stats_ret, 'descriptive_stats_ret', latex = True)


###########################################
# Estadisticos descriptivos - Volatilidad #
###########################################

# Proxy volatilidad |r|

volatility = abs(returns)

# Se genera la tabla 'descriptive_stats_vol'

descriptive_stats_vol = descriptiveStats(volatility)

# Se exporta a LaTeX

export(descriptive_stats_vol, 'descriptive_stats_vol', latex = True)


end = time.time()  # Fin cronometro
time_exe = str(timedelta(seconds = round(end - start))) # Tiempo de ejecucion

print('Ejecucion finalizada')
print('Tiempo de ejecución: ', time_exe)
