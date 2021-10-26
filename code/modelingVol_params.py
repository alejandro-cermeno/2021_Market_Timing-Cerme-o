'''
modelingVol_params.py

Alejandro Cermeño (09/2021)

Este codigo genera y exporta la tabla de coeficientes estimados
(modelingVol_params.xlsx y modelingVol_params.tex) para los modelos y series 
especificados. La diferencia entre ambos archivos está en la denominacion de la
significancia en letras (a,b,c) o en asteriscos (***,**,*). Además, se 
guarda el resumen de resultados de cada modelo en modelingVol_summary.txt.
Se requiere los paquetes time, numpy, pandas, datetime, itertools y arch.

Se omiten las tildes. Ver README.txt para informacion adicional.
'''

#############
# Librerias #
#############

import time
import numpy  as np
import pandas as pd

from datetime  import timedelta
from itertools import product 
from arch      import arch_model
from arch.__future__ import reindexing

start = time.time()  # Inicio cronometro 


#############
# Funciones #
#############

def price2ret(price):
  
  ret = (100 * ( np.log(price) - np.log(price.shift(1))))
  
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
  #elif especificacion == tarch_params:
  #  name2disp_ = 'TARCH'
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


def signif_letter(coef, pvalues):

  signif_letter = list()

  for p in range(len(pvalues)):
    if pvalues[p] <= 0.01:
      asterisco = '$^{a}$' # significativo al 1%                
    elif pvalues[p] <= 0.05:                
      asterisco = '$^{b}$'  # significativo al 5%                   
    elif pvalues[p] <= 0.1:
      asterisco = '$^{c}$'   # significativo al 10%                    
    else:
      asterisco = ''    # no significativo     

    valor = str(coef[p]) + asterisco # Se une el coeficiente y el asterisco

    signif_letter.append(valor)
  
  signif_letter = pd.DataFrame(signif_letter, index = coef.index)

  return signif_letter


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

path = 'https://git.io/J6hzs'

# Series del mercado bursatil

stocks  = pd.read_excel(path, sheet_name = 'stocks_raw', index_col = 0)
r_stocks = price2ret(stocks)

# Series del mercado cambiario
  
forex  = pd.read_excel(path, sheet_name = 'forex_raw', index_col = 0)
r_forex = price2ret(forex)

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
#tarch_params   = {'p':1, 'o':1, 'q':1, 'power':1.0}          # TARCH or ZARCH
#egarch_params  = {'p':1, 'q':1, 'o':1, 'vol': 'EGARCH'}      # EGARCH
aparch_params  = {'p':1, 'o':1, 'q':1, 'vol':'APARCH'}        # APARCH
figarch_params = {'p':1, 'q':1, 'power':2.0, 'vol':'FIGARCH'} # FIGARCH

variance_ops  = [arch_params, garch_params, grj_params, aparch_params,
                 figarch_params]

# Distributions

dist_ops     = ['normal', 't', 'skewt', 'ged']


# Almacenar resultados

modelingVol_params_excel = pd.DataFrame()
modelingVol_params_latex = pd.DataFrame()

series_n = returns.shape[1] # Numero de series

#######################################
# Resultados para cada serie y modelo #
#######################################

for market, mean, variance, dist in product(returns.columns, mean_ops, variance_ops, dist_ops):

    ts = returns[market].dropna()

    # Se verifica si el modelo especificado tiene solucion
    try:

        # Especificacion y estimacion del modelo
        mdl = arch_model_v2(ts, mean = mean, **variance, dist = dist, 
                            rescale = False).fit(disp='off')

        # Save summary models results

        with open("modelingVol_summary.txt", "a+") as file_object:
          file_object.write(str(mdl))
          file_object.write("\n")
          file_object.write("\n")


        # Resultados si el modelo tiene solucion
        
        # Media, varianza y distribución para mostrar en la tabla
        model_info = model_info_fun(ts, mean, variance, dist)

        # P-values
        pvalues = mdl.pvalues

        # Coeficientes
        coef = mdl.params.map('{:,.4f}'.format) # set 4 decimals precision

        coef_letter = pd.DataFrame(signif_letter(coef, pvalues))
        coef_ast = pd.DataFrame(signif_ast(coef, pvalues))

        # Log-verosimilutud
        loglik = pd.Series(mdl.loglikelihood, index = ['log-lik'])

        # Valor del criterio de informacion
        bic = pd.Series(mdl.bic, index = ['BIC'])

        # Singular Matrix (0 = el modelo pudo ser estimado)
        singular_matrix = pd.Series('0', index = ['Singular Matrix'])

        #  Tablas resultado 

        # Tabla para LaTeX (significancia con letras)
        #to_latex = pd.concat([model_info, coef_letter, bic, loglik, singular_matrix]) 
              
        # Tabla para Excel (significancia con asteriscos)
        to_excel = pd.concat([model_info,coef_ast, bic, loglik, singular_matrix]) 


    # En caso no tener solucion:
    
    except np.linalg.LinAlgError:
        
        # Media, varianza y distribución para mostrar en la tabla
        model_info = model_info_fun(ts, mean, variance, dist)
        
        # Singular Matrix (1 = el modelo no tuvo solución)
        singular_matrix = pd.Series('1', index = ['Singular Matrix'])

        # Valores para las otras columnas (parametros, BIC, etc) en blanco
        
        #  Tablas resultado 
        #to_latex = pd.concat([model_info, singular_matrix])
        to_excel = pd.concat([model_info, singular_matrix])
        
        
    # Se anexa la fila a las tablas de resultados (append)
      
    to_excel = pd.DataFrame(to_excel).T
    #to_latex = pd.DataFrame(to_latex).T
        
    modelingVol_params_excel = modelingVol_params_excel.append(to_excel, 
                                                                sort = False, 
                                                                ignore_index = True)
        
    #modelingVol_params_latex = modelingVol_params_latex.append(to_latex, 
    #                                                           sort = False,
    #                                                           ignore_index = True)


##########################################
# Organizacion de la tabla de resultados #
##########################################

# Remplazar NaN por espacios en blanco

modelingVol_params_excel = modelingVol_params_excel.fillna('')
#modelingVol_params_latex = modelingVol_params_latex.fillna('')

# Join columns: d with delta, nu with eta, Const with mu, beta[1] with beta

modelingVol_params_excel['delta'] = modelingVol_params_excel['delta'] +\
modelingVol_params_excel['d']
modelingVol_params_excel['mu'] = modelingVol_params_excel['mu'] +\
modelingVol_params_excel['Const']
modelingVol_params_excel['nu'] = modelingVol_params_excel['nu'] +\
modelingVol_params_excel['eta']

modelingVol_params_excel.drop(['Const', 'eta', 'd'], axis=1)

# Rename columns

modelingVol_params_excel.rename(columns = {'alpha[1]':'alpha',
                                           'beta[1]':'beta',
                                           'gamma[1]':'gamma'},
                                inplace = True)

# Columns order

col_order = ['serie', 'mean', 'variance', 'dist', 'mu', 'omega', 'alpha',
             'beta', 'gamma', 'delta', 'phi', 'nu', 'lambda', 'BIC', 'log-lik', 
             'Singular Matrix']
              
modelingVol_params_excel = modelingVol_params_excel[col_order]


#############################
# Exportacion de resultados #
#############################

  # A Excel
export(modelingVol_params_excel, 'modelingVol_params', Excel = True)

  # A LaTeX
#export(modelingVol_params_latex, 'modelingVol_params', LaTeX = True)


# Se finaliza el codigo 

end = time.time()                                       # Fin cronometro
time_exe = str(timedelta(seconds = round(end - start))) # Tiempo de ejecucion

print('Ejecucion finalizada')
print('Tiempo de ejecución: ', time_exe)
