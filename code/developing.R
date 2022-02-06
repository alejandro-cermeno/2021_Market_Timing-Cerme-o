library("openxlsx")
library("Rcpp")
library("rugarch")

# Data collection (all Excel sheets)
path = "demean_returns.xlsx"
sheets <- getSheetNames(path)
ret_all_demeaning <- lapply(sheets, 
                       function(sheet) read.xlsx(path, 
                                                 skipEmptyRows = TRUE, 
                                                 rowNames = TRUE, 
                                                 detectDates = TRUE, 
                                                 sheet = sheet))
names(ret_all_demeaning) <- sheets

# Store results
results = data.frame()

# For every demeaning filter
sheet <- "ma_1" # for sheet in sheets!!!!!
ret <- ret_all_filt[[sheet]]
series <- colnames(ret)

# For every serie
serie <- "SPBLPGPT" # for serie in series
ts <- na.omit(ret[serie])

# variance and distribution options
var_opt = c("sGARCH", "fiGARCH", "csGARCH", "gjrGARCH", "eGARCH", "apARCH")
dist_opt = c("norm", "snorm", "std", "sstd", "ged", "sged")


for (var in var_opt) {
  for (dist in dist_opt) {
    # Model identification
    spec = ugarchspec(variance.model = list(model = var), mean.model = list(armaOrder = c(0, 0), include.mean = TRUE), distribution.model = dist)
    
    # Model estimation
    fit = ugarchfit(data = ts,spec = spec)
    
    add2results = data.frame("deamining" = sheet,
                             "serie" = serie,
                             "variance" = var, 
                             "dist" = dist, 
                             as.list(fit@fit$coef),
                             "aic" = infocriteria(fit)[1],
                             "bic" = infocriteria(fit)[2], 
                             "loglik" = fit@fit$LLH)
    
    # Append estimated model to table
    results = dplyr::bind_rows(results, add2results) 
  }
}


# Results table configuration

out = c("sGARCH", "gjrGARCH", "eGARCH", "fiGARCH", "apARCH", "csGARCH",
        "norm", "snorm", "std", "sstd", "ged", "sged")
inn =  c("GARCH", "GJR", "EGARCH", "FIGARCH", "APARCH", "CSGARCH", "N",
         "skN", "t", "skt", "GED", "skGED")

n_changes <- if (length(inn) == length(out)) length(out) else stop()

for (i in 1:n_changes) {
  results[results == out[i]] <- inn[i]
}

forex <- c("ARS", "BRL", "CLP", "COP", "MXN", "PEN")
stock <- c("MERVAL", "IBOV", "IPSA", "IGBC", "MEXBOL", "SPBLPGPT")

results$market[(results$serie %in% forex)]  <- "forex"
results$market[(results$serie %in% stock)]  <- "stock"


colnames(results)

col_order <- c("deamining",
               "market",
               "serie",
               "variance",
               "dist",
               "mu", # constant in the mean equation
               "omega",
               "alpha1",
               "beta1",
               "gamma1"
               "aic",
               "bic",
               "loglik",
               "skew",
               "shape",
               "delta",
               
[15] "eta11"     "eta21"         )  


# Export to Excel
write.xlsx(results, file = "modeling_vol.xlsx")

