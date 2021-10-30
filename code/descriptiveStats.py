"""
descriptiveStats.py
Alejandro Cermeño (09/2021)
This code replicates Table 1: Descriptive Statistics for Stock and Forex Markets
Returns (descriptive_stats_ret.tex). It requires time, numpy, pandas, and datetime.

See README.txt for additional information.
"""

import time
import numpy as np
import pandas as pd

from datetime import timedelta

start = time.time()  # start stopwatch


def price2ret(price):
  ret = (100 * ( np.log(price) - np.log(price.shift(1))))
  return ret


def descriptiveStats(df):
    """The number of observations, mean, sd, minimum, maximum, skewness and 
    kurtosis are selected"""
  descriptive_stats = pd.concat([df.describe()\
                                 .loc[['count', 'mean', 'std', 'min', 'max']].T, 
                                 df.skew().rename('skew'), df.kurt().\
                                 rename('kurt')], axis=1).round(2)
  
  descriptive_stats['count'] = descriptive_stats['count'].astype(int)

  # Start date
  start_date = df.apply(lambda df: df.first_valid_index())
  start_date = pd.DataFrame(start_date, columns=['start date'])
  
  # End date
  end_date = df.apply(lambda df: df.last_valid_index())
  end_date = pd.DataFrame(end_date, columns=['end date'])

  descriptive_stats = pd.concat([start_date, end_date, 
                                 descriptive_stats], axis = 1)
  return descriptive_stats


def export(df, file_name, excel = None, latex = None):

  # To Excel
  if excel == True:
    df.to_excel(file_name + '.xlsx')  
    
  # To LaTeX
  if latex == True:
    latex_code = df.to_latex()
    with open(file_name + '.tex', 'w') as tex:
      tex.write(latex_code)


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


##############################################################
# Descriptive Statistics for Stock and Forex Markets Returns #
##############################################################

descriptive_stats_ret = descriptiveStats(returns)

# Results to LaTeX
export(descriptive_stats_ret, 'descriptive_stats_ret', latex = True) 


print('Execution completed')
print('Execution time;', time_exe)
