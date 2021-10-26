'''
backtestVaR.py
Alejandro Cermeño (09/2021)
El código aplica las pruebas de backtest de Kupiec (1995) y Christoffesen (1998)
en las proyecciones del VaR al 99%, 95% de confianza realizadas. También se calcula el MAE y MSE 
para las proyecciones de la volatilidad respecto a la volatilidad realizada (|r|). Se requieren las 
librerias pandas, datetime e itertools.
Se omiten las tildes. Ver README.txt para informacion adicional.
'''

# Librerias 

import numpy as np
import pandas as pd 
from scipy import stats
from datetime import timedelta
from itertools import product 
from sklearn.metrics import mean_absolute_error, mean_squared_error

#############
# Funciones #
#############

class varbacktest:
    r"""
    Implementation of backtest tests.

    Parameters
    ----------
    true : {ndarray, Series, None}
        The dependent variable
    pred : {ndarray, Series, None}
        The dependent variable
    lags : int, optional
        Default is 4.
    """

    def __init__(self, actual, forecast, alpha):
        self.index = actual.index
        self.actual = actual.values
        self.forecast = forecast.values
        self.alpha = alpha

    def hit_series(self):
        return (self.actual < self.forecast) * 1

    def number_of_hits(self):
        return self.hit_series().sum()

    def hit_rate(self):
        return self.hit_series().mean()

    def expected_hits(self):
        return self.actual.size * self.alpha

    def duration_series(self):
        hit_series = self.hit_series()
        hit_series[0] = 1
        hit_series[-1] = 1
        return np.diff(np.where(hit_series == 1))[0]

    def tick_loss(self, return_mean=True):
        loss = (self.alpha - self.hit_series()) * (self.actual - self.forecast)
        if return_mean:
            return loss.mean()
        else:
            return loss

    def cc(self):
        """Likelihood ratio framework of Christoffersen (1998)"""
        hits = self.hit_series()   # Hit series
        tr = hits[1:] - hits[:-1]  # Sequence to find transitions

        # Transitions: nij denotes state i is followed by state j nij times
        n01, n10 = (tr == 1).sum(), (tr == -1).sum()
        n11, n00 = (hits[1:][tr == 0] == 1).sum(), (hits[1:][tr == 0] == 0).sum()

        # Times in the states
        n0, n1 = n01 + n00, n10 + n11
        n = n0 + n1

        # Probabilities of the transitions from one state to another
        p01, p11 = n01 / (n00 + n01), n11 / (n11 + n10)
        p = n1 / n

        if n1 > 0:
            # Unconditional Coverage
            uc_h0 = n0 * np.log(1 - self.alpha) + n1 * np.log(self.alpha)
            uc_h1 = n0 * np.log(1 - p) + n1 * np.log(p)
            uc = -2 * (uc_h0 - uc_h1)

            # Independence
            ind_h0 = (n00 + n01) * np.log(1 - p) + (n01 + n11) * np.log(p)
            ind_h1 = n00 * np.log(1 - p01) + n01 * np.log(p01) + n10 * np.log(1 - p11)
            if p11 > 0:
                ind_h1 += n11 * np.log(p11)
            ind = -2 * (ind_h0 - ind_h1)

            # Conditional coverage
            cc = uc + ind

            PVuc = '{:.15f}'.format(1 - stats.chi2.cdf(uc, 1))
            PVcci = 1 - stats.chi2.cdf(ind, 1)
            PVcc = '{:.15f}'.format(1 - stats.chi2.cdf(cc, 2))
        
        else:
            PVuc = np.nan
            PVcc = np.nan

        # Assign names
        #df.columns = ["Statistic", "p-value"]
        #df.index = ["Unconditional", "Independence", "Conditional"]
        return pd.Series([PVuc, PVcc], index=["PVuc", "PVcc"])

    def dq(self, hit_lags=4, forecast_lags=1):
        """Dynamic Quantile Test (Engle & Manganelli, 2004)"""
        try:
            hits = self.hit_series()
            p, q, n = hit_lags, forecast_lags, hits.size
            pq = max(p, q - 1)
            y = hits[pq:] - self.alpha  # Dependent variable
            x = np.zeros((n - pq, 1 + p + q))
            x[:, 0] = 1  # Constant

            for i in range(p):  # Lagged hits
                x[:, 1 + i] = hits[pq-(i+1):-(i+1)]

            for j in range(q):  # Actual + lagged VaR forecast
                if j > 0:
                    x[:, 1 + p + j] = self.forecast[pq-j:-j]
                else:
                    x[:, 1 + p + j] = self.forecast[pq:]

            beta = np.dot(np.linalg.inv(np.dot(x.T, x)), np.dot(x.T, y))
            lr_dq = np.dot(beta, np.dot(np.dot(x.T, x), beta)) / (self.alpha * (1-self.alpha))
            PVdq = '{:.15f}'.format(1 - stats.chi2.cdf(lr_dq, 1+p+q))

        except:
            lr_dq, PVdq = np.nan, np.nan

        return pd.Series(PVdq, index=["PVdq"])


def pct_fails(returns, VaR):

    n_fails = ((returns.values < VaR.values) * 1).sum()

    return n_fails / len(VaR)


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
# CHECK: IPSA
df = pd.read_excel('/content/BRL_forecastVolVaR_5553_OOS.xlsx')

df['VaR_1'] = df['VaR_1'].apply(lambda x: x*-1)
df['VaR_5'] = df['VaR_5'].apply(lambda x: x*-1)

####################
# Especificaciones #
####################

# Media
mean_ops       = ['Zero', 'Constant', 'AR']
#mean_ops = df['mean'].unique()

# Varianza
arch_params    = {'vol': 'ARCH'}
garch_params   = {'p':1, 'q':1, 'vol':'GARCH'}
grj_params     = {'p':1, 'o':1, 'q':1, 'vol':'GARCH'}
egarch_params  = {'p': 1, 'q': 1, 'o': 1, 'vol': 'EGARCH'}
aparch_params  = {'p':1, 'o':1, 'q':1, 'power': 2.0, 'vol':'GARCH'}
#figarch_params = {'p':1, 'q':1, 'power': 2.0, 'vol':'FIGARCH'}
variance_ops   = [arch_params, garch_params, grj_params, egarch_params, 
                  aparch_params] # figarch_params

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

  returns = filtered['mean_true']
  vol_true = filtered['vol_true']
  vol_pred = filtered['vol_pred']

  # Tiempo de ejecucion
  #time_exe = sum(list(map(float, filtered['time'].values)))

  # MSE (vol_true vs vol_pred)
  mse = mean_squared_error(vol_true.values, vol_pred.values)

  # MAE (vol_true vs vol_pred)
  mae = mean_absolute_error(vol_true.values, vol_pred.values)


  ######################################
  # Metricas proyeccion VaR (backtest) #
  ######################################

  VaR_ops     = ['VaR_1', 'VaR_5']
  conf_lvl_ops = [0.01, 0.05]

  # Para cada nivel de confianza 

  for i in range(len(VaR_ops)):

    VaR = filtered[VaR_ops[i]]
    conf_lvl = conf_lvl_ops[i]

    # Porcentaje de fallas VaR
    pct_fails_ = pct_fails(returns, VaR)

    # Pruebas de backtest

    bt = varbacktest(actual=returns, forecast=VaR, alpha=conf_lvl)
    uc = bt.cc()['PVuc'] # Kupiec (1995)
    cc = bt.cc()['PVcc'] # Christoffersen (1998)
    dq = bt.dq()['PVdq'] # Engle and Manganelli (2004)

    #######################
    # Tabla de resultados #
    #######################


    serie = pd.Series(df['serie'], name = df['serie'].iloc[1])
    model_info2t = model_info_fun(serie, mean, variance, dist)

    mae2t = pd.Series(mae, index = ['MAE'])
    mse2t = pd.Series(mse, index = ['MSE'])
    VaR_lvl2t = pd.Series(conf_lvl, index = ['VaR_lvl'])
    pct_fails_2t = pd.Series(pct_fails_, index = ['pct_fails'])
    uc2t = pd.Series(uc, index = ['uc'])
    cc2t = pd.Series(cc, index = ['cc'])
    dq2t = pd.Series(dq, index = ['dq'])

    #time_exe2t = pd.Series(time_exe, index = ['time_exe'])

    to_backtestVaR = pd.concat([model_info2t, mae2t, mse2t, VaR_lvl2t, 
                                pct_fails_2t, uc2t, cc2t, dq2t])
    to_backtestVaR = pd.DataFrame(to_backtestVaR).T

    backtestVaR = backtestVaR.append(to_backtestVaR)

# Se exportan los resultados
export(backtestVaR, 'backtestVaR_' + df['serie'][1], Excel = True)

print('Archivo backtestVaR_' + df['serie'][1] + '.xlsx guardado con éxito')

# Se finaliza el código
print('Ejecución finalizada')

backtestVaR
