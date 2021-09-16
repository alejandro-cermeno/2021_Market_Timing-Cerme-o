'''
backtest_VaR.R

Alejandro Cermeño (09/2021)

El código aplica las pruebas de backtest de Kupiec (1995), Christoffesen (1998)
y Engle and Manganelli (2004) en las proyecciones del VaR al 99%, 95% y 90% de
confianza realizadas. También se calcula el MAE y MSE para las proyecciones de la 
volatilidad respecto a la volatilidad realizada (|r|). Se requieren las 
librerias readxl, tidyverse y GAS.

Se omiten las tildes. Ver README.txt para informacion adicional.
'''

#############
# Librerias #
#############

library('readxl')
library('dplyr')
library("stringr")
library('hms')
library('GAS')
library('itertools')


#############
# Funciones #
#############

set_format <- function(vals) {
  out <- format(round(vals, 4), nsmall = 4)
  return(out) # Se fijan 4 decimales a mostrarse
}

mean_square_error <- function(true, pred) {
  res <- mean((true - pred)^2)
  return(set_format(res))
}

mean_absolute_error <- function(true, pred) {
  res <- mean( abs(true - pred))
  return(set_format(res))
}


#########################################
# Obtencion de datos y especificaciones #
#########################################

forecastVolVaR <- read_excel('BRL_forecastVolVaR_49_OOS.xlsx')

# Nombre serie

serie        = as.character(unique(unlist(forecastVolVaR$serie)))

# Opciones modelo

mean_opc     = unique(unlist(filtered$mean))
variance_opc = unique(unlist(filtered$variance))
dist_opc     = unique(unlist(filtered$dist))

# Opciones VaR

VaR_opc      = c('VaR_1', 'VaR_5', 'VaR_10')
VaR_lvl_opc  = c(99, 95, 90)
conf_lvl_opc = c(0.01, 0.05, 0.1)

# Almacenamiento de resultados

backtest_VaR = data.frame()

# Para los resultados de cada modelo

for (mean_ in mean_opc) {
    
  for (variance_ in variance_opc) {
      
    for (dist_ in dist_opc) {
      
      filtered = filter(forecastVolVaR, mean == mean_ & variance == variance_ & dist == dist_)
      
      # Retornos
      
      returns  = filtered$mean_true
      
      # MSE (vol_true vs vol_pred)
      MSE = mean_square_error(filtered$vol_true, filtered$vol_pred)
      
      # MAE (vol_true vs vol_pred)
      MAE = mean_absolute_error(filtered$vol_true, filtered$vol_pred) 
      
      # Tiempo ejecucion
      time_exe = unlist(filtered['time'])
      time_exe = as_hms(sum(as.numeric(time_exe)))
      
      
############
# Backtest #
############
      
      # Para cada nivel de confianza del VaR
      for (i in 1:length(VaR_opc)){
        
        VaR = unlist(filtered[VaR_opc[i]])
        conf_lvl = conf_lvl_opc[i]
        VaR_lvl = VaR_lvl_opc[i]
        
        # Calculo pruebas POF, CC y DQ
        
        backtest = BacktestVaR(returns, VaR, conf_lvl) 
        
        # P-values de las pruebas
        
        POF = set_format(backtest$LRuc[2])
        CC = set_format(backtest$LRcc[2])
        DQ = set_format(backtest$DQ$pvalue)
        
        # Resultados
        
        to_append = data.frame(serie, mean_, variance_, dist_, VaR_lvl, POF,
                               CC, DQ, MSE, MAE, time_exe)
        
        # Adicion de los resutados del modelo y VaR en específico a la tabla de resultados
        
        backtest_VaR = rbind(backtest_VaR, to_append)
        
      }
    }
  }
}

# Se reinicia el index

row.names(backtest_VaR) <- NULL

print('Ejecucion finalizada')