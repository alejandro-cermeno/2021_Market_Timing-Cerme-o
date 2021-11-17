"""
Ardia D, Boudt K, Catania L (2019). “Generalized Autoregressive Score Models in R: The GAS Package.” Journal of Statistical Software, 88(6), 1–28.
"""
# Funcion #
###########

vY = returns
vVaR = VaR
dTau = 0.05

vY = as.numeric(vY)
vVaR = as.numeric(vVaR)

Hit = HitSequence(vY, vVaR)
UC = Kupiec(Hit, dTau)
BacktestVaR <- function(data, VaR, alpha, cLags = 4L) {
  
  vY = data
  vVaR = VaR
  dTau = alpha
  
  vY = as.numeric(vY)
  vVaR = as.numeric(vVaR)
  
  Hit = HitSequence(vY, vVaR)
  UC = Kupiec(Hit, dTau)
  CC = Christoffersen(Hit, dTau)
  DQ = DQOOStest(vY, vVaR, dTau, cLags)
  AE = ActualOverExpected(Hit, dTau)
  AD = AbsoluteDeviation(Hit, vY, vVaR)
  Loss = QLoss(vY, vVaR, dTau)
  
  lOut = data.frame("VaR_lvl" = (1-dTau),  "obs" = UC["obs"], 
                    "num_hits" = UC["num_hits"], "pct_hits" = UC["pct_hits"],
                    "LRuc" = UC["LRuc"], "PVuc" = UC["PVuc"], 
                    "LRcci" = CC["LRcci"],"PVcci" = CC["PVcci"], 
                    "LRcc" = CC["PVcc"], "PVcc" = CC["PVcc"], "DQ" = DQ["DQ"], 
                    "PVdq" = DQ["PVdq"])
  return(lOut)
}


DQOOStest <- function(y, VaR, tau, cLags) {
  
  cT = length(y)
  vHit = numeric(cT)
  vHit[y < VaR] = 1 - tau
  vHit[y > VaR] = -tau
  
  vConstant = rep(1, (cT - cLags))
  vHIT = vHit[(cLags + 1):cT]
  vVaRforecast = VaR[(cLags + 1):cT]
  mZ = matrix(0, cT - cLags, cLags)
  vY2_lag = y[cLags:(cT - 1)]^2
  
  for (st in 1:cLags) {
    
    mZ[, st] = vHit[st:(cT - (cLags + 1L - st))]
    
  }
  
  mX = cbind(vConstant, vVaRforecast, mZ, vY2_lag)
  
  dDQstatOut = (t(vHIT) %*% mX %*% MASS::ginv(t(mX) %*% mX) %*% t(mX) %*% (vHIT))/(tau * (1 - tau))
  
  dDQpvalueOut = 1 - pchisq(dDQstatOut, ncol(mX))
  
  DQ = c(dDQstatOut, dDQpvalueOut)
  names(DQ) = c("DQ", "PVdq")
  return(DQ)
}

HitSequence <- function(returns_X, VaR_X) {
  N = length(returns_X)
  Hit_X = numeric(N)
  Hit_X[which(returns_X <= VaR_X)] = 1L
  return(Hit_X)
}

Kupiec <- function(Hit, tau) {
  N = length(Hit)
  x = sum(Hit)
  rate = x/N
  test = -2 * log(((1 - tau)^(N - x) * tau^x)/((1 - rate)^(N - x) * rate^x))
  if (is.nan(test))
    test = -2 * ((N - x) * log(1 - tau) + x * log(tau) - (N - x) * log(1 - rate) - x * log(rate))
  # threshold = qchisq(alphaTest, df = 1)
  pvalue = 1 - pchisq(test, df = 1)
  
  LRpof = c(N, x, rate, test, pvalue)
  names(LRpof) = c("obs", "num_hits", "pct_hits", "LRuc", "PVuc")
  return(LRpof)
}

Christoffersen <- function(Hit, tau) {
  n00 = n01 = n10 = n11 = 0
  N = length(Hit)
  for (i in 2:N) {
    if (Hit[i] == 0L & Hit[i - 1L] == 0L)
      n00 = n00 + 1
    if (Hit[i] == 0L & Hit[i - 1L] == 1L)
      n01 = n01 + 1
    if (Hit[i] == 1L & Hit[i - 1L] == 0L)
      n10 = n10 + 1
    if (Hit[i] == 1L & Hit[i - 1L] == 1L)
      n11 = n11 + 1
  }
  pi0 = n01/(n00 + n01)
  pi1 = n11/(n10 + n11)
  pi = (n01 + n11)/(n00 + n01 + n10 + n11)
  LRind = -2 * log(((1 - pi)^(n00 + n10) * pi^(n01 + n11))/((1 - pi0)^n00 * pi0^n01 * (1 - pi1)^n10 *
                                                              pi1^n11))
  if (is.nan(LRind))
    LRind = -2 * ((n00 + n10) * log(1 - pi) + (n01 + n11) * log(pi) - n00 * log(1 - pi0) - n01 *
                    log(pi0) - n10 * log(1 - pi1) - n11 * log(pi1))
  LRpof = Kupiec(Hit, tau)["Test"]
  LRcc = LRpof + LRind
  PVcci = 1 - pchisq(LRind, df = 1)
  PVcc = 1 - pchisq(LRcc, df = 2L)
  LRcc = c(LRind, PVcci, LRcc, PVcc)
  names(LRcc) = c("LRcci", "PVcci", "LRcc", "PVcc")
  return(LRcc)
}

ActualOverExpected <- function(Hit, tau) {
  N = length(Hit)
  x = sum(Hit)
  Actual = x
  AovE = Actual/(tau * N)
}

AbsoluteDeviation <- function(Hit, returns_X, VaR_X) {
  series = abs(VaR_X - returns_X)
  series = series[which(Hit == 1L)]
  ADmean = mean(series)
  ADmax = max(series)
  
  out = c(ADmean, ADmax)
  names(out) = c("ADmean", "ADmax")
  return(out)
}

QLoss <- function(vY, vVaR, dTau) {
  vHit = HitSequence(vY, vVaR)
  vLoss = (vY - vVaR) * (dTau - vHit)
  dLoss = mean(vLoss)
  
  return(list(Loss = dLoss, LossSeries = vLoss))
}

########
# Test #
########

# data collection
path = "https://git.io/J1clf"
#df <- read.xlsx(path, skipEmptyRows = FALSE)

# specifications
VaR_ops = c("VaR_1", "VaR_5")
conf_lvl_ops = c(0.01, 0.05)
returns = df$mean_true

# backtest

for(i in 1 : length(VaR_ops)) {
  VaR = df[[VaR_ops[i]]]
  print(BacktestVaR(returns, VaR, conf_lvl_ops[i]))
}

#   VaR_lvl  obs  num_hits pct_hits LRuc PVuc LRcci PVcci LRcc PVcc DQ   PVdq
#   0.99     1703 166      0.09748  Inf  0    294.5 0     NA    NA  7609 0
#
#   VaR_lvl  obs  num_hits pct_hits LRuc PVuc LRcci PVcci LRcc PVcc DQ   PVdq
#   0.95     1703 284      0.1668   312  0    Inf   0     NA   NA   1676 0
