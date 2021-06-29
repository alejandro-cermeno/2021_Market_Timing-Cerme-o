# En desarrollo...

The following files replicate the results in 'Title in progress'

* returns_figure_descriptive.ipynb replicates Figure 1 and Table 1

* models.ipynb replicates Table 2




#=========
Referencia:
%The following files replicate the empirical results in: 
%Chen, Rogoff and Rossi,  
%'Can Exchange Rates Forecast Commodity Prices?' 
%version June 2009

EW7redo.m                           replicates  Tables 1 2 3 4 and 8 (includes ENDOG tests and RWWD) as well as Table C1
EWforecastcomb1GlobalCommREDO.m     replicates  Figures 1 & 2 and Table 5 (global comm prices in multiv regressions and forecast combin)
EW7globalcommpricesREDO.m           replicates  Table 6 (global commodity prices for individual exchange rates)
  Note: Table 6b obtained by using tableb=1 in both EW7globalcommpricesREDO.m and EWforecastcomb1GlobalCommREDO.m (for the multiv forecasts)
EWforecastcomb1GlobalCommREDOfwdAIGALL.m    replicates Figure 5 for DJ-AIG (change lines 67-68 in  plotforectestshacfwd.m to change the forward regression model)
EW7crossratesREDO.m                 replicates  Table 7a (crossrates for NEER) 
EW7crossratesUKredo.m               replicates  Table 7b (crossrates for the U.K) 
EWforecastChileREDO.m               replicates  Figure 3
EW7forwards.m                       replicates Table 9 
EW7redoALLmacroREDO.m                   replicates Table C2: results for OTHER macro variables
EW7redoOVERTIME.m                   replicates Figure C1: (OOS tests over time)

In all files, set lagdep=1 for the AR(p) benchmark 
and set lagdep=0 for the RW benchmark in OOS regression 
