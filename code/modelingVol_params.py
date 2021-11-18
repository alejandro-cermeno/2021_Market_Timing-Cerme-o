'''
modelingVol_params.py
Alejandro Cermeño (09/2021)

This code generates and exports the Table 2: Estimated Parameters for daily Latin
American Stock and Forex Markets Return (modelingVol_params.xlsx). In addition, 
the summary of results for each model is saved in modelingVol_summary.txt. 
Requires time, numpy, pandas, datetime, itertools and arch.

See README.txt for additional information.

'''

import time
import numpy  as np
import pandas as pd
from datetime  import timedelta
from itertools import product 
from arch      import arch_model
from arch.__future__ import reindexing

start = time.time()  # start stopwatch  

#############
# Functions #
#############

def price2ret(price):
  return (100 * ( np.log(price) - np.log(price.shift(1))))

def export(df, file_name, excel = None, latex = None):

  # To Excel
  if excel == True:
    df.to_excel(file_name + '.xlsx')  
        
  # To LaTeX
  if latex == True:
    latex_code = df.to_latex()
    with open(file_name + '.tex', 'w') as tex:
      tex.write(latex_code)


def signif_ast(coef, pvalues):
    """Sets the confidence level in asterisks."""
    
    signif_ast = list()

    for p in range(len(pvalues)):
      if pvalues[p] <= 0.01:
        asterisks = '***' # significant at 1%             
      elif pvalues[p] <= 0.05:                
        asterisks = '**'  # significant at 5%                   
      elif pvalues[p] <= 0.1:
        asterisks = '*'   # significant at 10%                    
      else:
        asterisks = ''    # not significant     

      value = str(coef[p]) + asterisks # joins the coefficient and the asterisks

      signif_ast.append(value)

    signif_ast = pd.DataFrame(signif_ast, index = coef.index)
  
    return signif_ast  


def signif_letter(coef, pvalues):
    """Sets the confidence level in asterisks."""

    signif_letter = list()

    for p in range(len(pvalues)):
      if pvalues[p] <= 0.01:
        asterisks = '$^{a}$' # significant at 1%                
      elif pvalues[p] <= 0.05:                
        asterisks = '$^{b}$'  # significant at 5%                   
      elif pvalues[p] <= 0.1:
        asterisks = '$^{c}$'   # significant at 10%                
      else:
        asterisks = ''    # not significant       

      value = str(coef[p]) + asterisks # joins the coefficient and the asterisks

      signif_letter.append(value)
    
    signif_letter = pd.DataFrame(signif_letter, index = coef.index)

    return signif_letter


def name2disp(especificacion):
  """For each specification of the mean, variance, or distribution, set 
  the name of the specification to be displayed."""

  # Mean specification
  if especificacion == 'Zero':
    name2disp_ = 'Cero'
  elif especificacion == 'Constant':
    name2disp_ = 'Constante'
  elif especificacion == 'AR':
    name2disp_ = 'AR'

    # Variance model
  elif especificacion == arch_params:
    name2disp_ = 'ARCH'
  elif especificacion == garch_params:
    name2disp_ = 'GARCH'
  elif especificacion == grj_params:
    name2disp_ = 'GJR'
  #elif especificacion == tarch_params: # Not included
  #  name2disp_ = 'TARCH'
  #elif especificacion == egarch_params: # Not included
  #  name2disp_ = 'EGARCH'
  elif especificacion == aparch_params:
    name2disp_ = 'APARCH'
  elif especificacion == figarch_params:
    name2disp_ = 'FIGARCH'
   # Error distributions
  elif especificacion == 'normal':
    name2disp_ = 'N'
  elif especificacion == 't':
    name2disp_ = 't'
  elif especificacion == 'skewt':
    name2disp_ = 'skt'
  elif especificacion == 'ged':
    name2disp_ = 'GED'

  else:
    name2disp_ = 'Invalid specification'

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

######################################
# Data collection and specifications #
######################################

path = "https://git.io/JX83R"

prices  = pd.read_excel(path, index_col = 0)
returns = 100 * (np.log(prices) - np.log(prices.shift(1)))

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

modelingVol_params_excel = pd.DataFrame()
series_n = returns.shape[1] 

#####################################
# Results for each series and model #
#####################################

for market, mean, variance, dist in product(returns.columns, mean_ops, variance_ops, dist_ops):

    ts = returns[market].dropna()

    # It is verified if the specified model has a solution
    try:

        # Specification and estimation of the model
        mdl = arch_model_v2(ts, mean = mean, **variance, dist = dist, 
                            rescale = False).fit(disp='off')

        # Results if the model has solution
        
        # Save summary models results
        with open("modelingVol_summary.txt", "a+") as file_object:
          file_object.write(str(mdl))
          file_object.write("\n")
          file_object.write("\n")
        
        # Mean, variance and distribution to display in the table
        model_info = model_info_fun(ts, mean, variance, dist)

        # P-values
        pvalues = mdl.pvalues

        # Coefficients
        coef = mdl.params.map('{:,.4f}'.format) # set 4 decimals precision

        coef_letter = pd.DataFrame(signif_letter(coef, pvalues))
        coef_ast = pd.DataFrame(signif_ast(coef, pvalues))

        # Log-like
        loglik = pd.Series(mdl.loglikelihood, index = ['log-lik'])

        # Information criterion value
        bic = pd.Series(mdl.bic, index = ['BIC'])

        # Singular Matrix (0 = the model could be estimated)
        singular_matrix = pd.Series('0', index = ['Singular Matrix'])


        # Table for Excel (significancia con asteriscos)
        to_excel = pd.concat([model_info,coef_ast, bic, loglik, singular_matrix]) 


    # In case there is no solution:
  
    except np.linalg.LinAlgError:
        
        # Mean, variance and distribution to display in the table
        model_info = model_info_fun(ts, mean, variance, dist)
        
        # Singular Matrix (1 = the model had no solution)
        singular_matrix = pd.Series('1', index = ['Singular Matrix'])

        # Values for the other columns (parameters, BIC, etc) blank
        
        #  Results table
        to_excel = pd.concat([model_info, singular_matrix])
        
        
    # The row is appended to the results tables (append)      
    to_excel = pd.DataFrame(to_excel).T
        
    modelingVol_params_excel = modelingVol_params_excel.append(to_excel, 
                                                                sort = False, 
                                                                ignore_index = True)

####################################
#Organization of the results table #
####################################

# Replace NaN with blanks
modelingVol_params_excel = modelingVol_params_excel.fillna('')

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



# Export of results 
export(modelingVol_params_excel, 'modelingVol_params', excel = True)


# The code is finalized
end = time.time()                                       # End of timer
time_exe = str(timedelta(seconds = round(end - start))) # Execution time

print('Execution completed')
print('Execution time: ', time_exe)
