'''
modelingVol_params.py

Alejandro Cermeño (09/2021)

Este codigo genera y exporta la tabla de coeficientes estimados
(modelingVol_params.xlsx y modelingVol_params.tex) para los modelos y series 
especificados. La diferencia entre ambos archivos está en la denominacion de la
significancia en letras (a,b,c) o en asteriscos (***,**,*). Se requiere los 
paquetes time, numpy, pandas, datetime, itertools y arch.

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

variance_ops     = [arch_params, garch_params, grj_params, egarch_params, 
                    aparch_params]

# Distribuciones

dist_ops      = ['normal', 't', 'skewt', 'ged']

# Almacenar resultados

modelingVol_params_excel = pd.DataFrame()
modelingVol_params_latex = pd.DataFrame()

series_n = returns.shape[1] # Numero de series

#######################################
# Resultados para cada serie y modelo #
#######################################

for market, mean, variance, dist in product(returns.columns, mean_ops, variance_ops, dist_ops):

    ts = returns[market].dropna()

    # Se verifica si el modelo especificado tiene solucion (try)
    try:

        # Especificacion y estimacion del modelo
        mdl = arch_model(ts, mean = mean, **variance, dist = dist).fit(disp='off')


        # Resultados si el modelo tiene solucion
        
        # Media, varianza y distribución para mostrar en la tabla
        model_info = model_info_fun(ts, mean, variance, dist)

        # P-values
        pvalues = round(mdl.pvalues, 3)

        # Coeficientes
        coef = round(mdl.params, 3)

        coef_letter = pd.DataFrame(signif_letter(coef, pvalues))
        coef_ast = pd.DataFrame(signif_ast(coef, pvalues))

        # Log-verosimilutud
        loglik = round(mdl.loglikelihood, 3)
        loglik = pd.Series(loglik, index = ['log-lik'])

        # Valor del criterio de informacion
        bic = round(mdl.bic, 3)
        bic = pd.Series(bic, index = ['BIC'])

        # Singular Matrix (0 = el modelo pudo ser estimado)
        singular_matrix = pd.Series('0', index = ['Singular Matrix'])

        #  Tablas resultado 

        # Tabla para LaTeX (significancia con letras)
        to_latex = pd.concat([model_info, coef_letter, bic, loglik, singular_matrix]) 
              
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
        to_latex = pd.concat([model_info, singular_matrix])
        to_excel = pd.concat([model_info, singular_matrix])
        
        
    # Se anexa la fila a las tablas de resultados (append)
      
    to_excel = pd.DataFrame(to_excel).T
    to_latex = pd.DataFrame(to_latex).T
        
    modelingVol_params_excel = modelingVol_params_excel.append(to_excel, 
                                                                   sort = False, 
                                                                   ignore_index = True)
        
    modelingVol_params_latex = modelingVol_params_latex.append(to_latex, 
                                                                   sort = False,
                                                                   ignore_index = True)


#######################################
# Formato y exportacion de resultados #
#######################################

# Remplazar NaN por espacios en blanco

modelingVol_params_excel = modelingVol_params_excel.fillna('')
modelingVol_params_latex = modelingVol_params_latex.fillna('')

# Ordenar las columnas

col_names = list(modelingVol_params_excel.columns)

right_columns =  ['BIC',  'log-lik', 'Singular Matrix']
left_columns = [l for l in col_names if l not in right_columns]

col_order = left_columns + right_columns

modelingVol_params_excel = modelingVol_params_excel[col_order]
modelingVol_params_latex = modelingVol_params_latex[col_order]

# Exportar tablas de resultados 

  # A Excel
export(modelingVol_params_excel, 'modelingVol_params', Excel = True)

  # A LaTeX
export(modelingVol_params_latex, 'modelingVol_params', LaTeX = True)


# Se finaliza el codigo 

end = time.time()                                       # Fin cronometro
time_exe = str(timedelta(seconds = round(end - start))) # Tiempo de ejecucion

print('Ejecucion finalizada')
print('Tiempo de ejecución: ', time_exe)