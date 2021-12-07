#install.packages('openxlsx')
#install.packages('GAS')
#install.packages('rugarch')
#install.packages("writexl")

library("openxlsx")
library("GAS")
library("rugarch")
library("writexl")

#
# data collection
#

# for every forecasted VaR serie
id_ops = c("1_XdPNEUjK-HGkZ_5W03UdCqQ3tlESlCw",  # ARS
           "1_Y2SFQ8q1qYzA9TZkatBrWGu5Qgu9MgA",  # BRL
           "1_PcbSMaqsy7n0MX4LJAfaXApvuvw7JII",  # IBOV
           "1bwPZ5_wzJbM0lA9HVHFLSFfNqGDNKfmV",  # IPSA
           "1_Uw9Fs21ARZdceeXP7uGZTvttn20Yw51",  # MEXBOL
           "1bc0CUUaSfuSO7FuHTdTJxRxOQlhGwN8e",  # PEN
           "1bx7WjCSCnlMeGT7YZTORIeneCHUrECRo"   # SPBLPGPT
)
drive_down = "https://drive.google.com/uc?export=download&id="

# for stored results
results = data.frame()

for (id in id_ops) {
  path = paste(drive_down, id, sep="")
  df <- read.xlsx(path, skipEmptyRows = FALSE)
  
  #
  # Specifications
  #
  
  mean_ops = unique(df$mean)
  variance_ops = unique(df$variance)
  dist_ops = unique(df$dist)
  VaR_ops = c("VaR_1", "VaR_5")
  sig_lvl_ops = c(0.01, 0.05)
  
  # Backtest for each model mean-variance-dist
  for (mean in mean_ops) {
    for (variance in variance_ops) {
      for (dist in dist_ops) {
        
        filtered = df[(df$mean == mean) & (df$variance == variance) & (df$dist == dist),]
        
        returns = filtered$mean_true
        
        for(i in 1 : length(VaR_ops)) {
          
          conf_lvl = 1 - sig_lvl_ops[i]
          VaR = filtered[[VaR_ops[i]]]
          
          # Generalized Autoregressive Score, GAS package of Ardia et al. (2018)
          gas = BacktestVaR(returns, VaR, sig_lvl)
          
          # R Univariate GARCH, rugarch package of Ghalanos (2020)
          rugarch = VaRTest(alpha = sig_lvl, returns, VaR, conf.level = conf_lvl)
          
          # Results table
          add = data.frame("serie" = unique(df$serie),
                           "mean" = mean,
                           "variance" = variance, 
                           "dist" = dist,
                           "VaR_lvl" = conf_lvl,
                           "obs" = length(returns), 
                           "num_hits" = rugarch$actual.exceed,
                           "pct_hits" = rugarch$actual.exceed / length(returns),
                           "actual_expected" = gas$AE,
                           "gas_pvUC" = gas$LRuc["Pvalue"], 
                           "gas_pvCC" = gas$LRcc["Pvalue"],
                           "gas_pvDQ" = as.numeric(unlist(gas$DQ["pvalue"])),
                           "ru_pvUC" = rugarch$uc.LRp,
                           "ru_pvCC" = rugarch$cc.LRp,
                           "ru_UC" = rugarch$uc.Decision,
                           "ru_CC" = rugarch$cc.Decision,
                           "H0_UC" = rugarch$uc.H0,
                           "H0_CC" = rugarch$cc.H0)
          results <- rbind(results, add)
        }
      }
    }
 }
}
