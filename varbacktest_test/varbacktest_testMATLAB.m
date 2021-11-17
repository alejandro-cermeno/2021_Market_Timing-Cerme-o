% MathWorks, (2012). Risk Management Toolbox: User's Guide (R2021b).

format long 

% Data collection
URL = ["https://git.io/J1clf"];
filename = "toyserie.xlsx";
urlwrite(URL, filename);
df = readtimetable(filename);

% specifications
df.VaR_1 = df.VaR_1.*-1;
df.VaR_5 = df.VaR_5.*-1;
VaR_ops = ["VaR_1", "VaR_5"];
conf_lvl_ops = [0.99, 0.95];
returns = df.mean_true;

% backtest
for i = 1 : length(VaR_ops)
    VaR = df.(i + 1);
    vbt = varbacktest(returns, VaR, 'VaRLevel', conf_lvl_ops(i));
    UC = pof(vbt);
    obs = UC.Observations;
    num_hits = UC.Failures;
    pct_fails = num_hits/obs;
    TL = tl(vbt).TL;
    LRuc = UC.LRatioPOF;
    PVuc = UC.PValuePOF;
    LRcci = cci(vbt).LRatioCCI;
    PVcci = cci(vbt).PValueCCI;
    LRcc = cc(vbt).LRatioCC;
    PVcc = cc(vbt).PValueCC;
    [DQ, PVdq] = dq(returns, VaR, conf_lvl_ops(i));
    
    results = table(conf_lvl_ops(i), obs, num_hits, pct_fails, TL, LRuc, PVuc, LRcci, ...
                    PVcci, LRcc, PVcc, DQ, PVdq, 'VariableNames', {'VaR_lvl', ...
                    'obs', 'num_hits', 'pct_fails', 'TL', 'LRuc', 'PVuc', 'LRcci', 'PVcci', ...
                    'LRcc', 'PVcc', 'DQ', 'PVdq'});
    disp(results)
end


function [DQ, PVdq] = dq(returns, VaR, sigVaR)
% Dynamic quantile test (DQ) of Engle and Manganelli (2004)
    L = 4;
    T = size(returns,1);
    Hit = (returns < VaR) - sigVaR;
    yy = Hit(L+1:T);
    Z = nan(T-L, L+2);
    for t = 1:T-L
        Z(t,:) = [1 Hit(L+t-1:-1:t)' VaR(L+t-1)];
    end
    
    deltahat = Z\yy;
    DQ      = deltahat'*(Z'*Z)*deltahat/(sigVaR*(1-sigVaR));
    PVdq    = 1 - chi2cdf(DQ, L+2);
end

% Output
%
%    VaR_lvl    obs     num_hits        pct_fails         TL           LRuc                  PVuc                  LRcci                 PVcci                  LRcc                  PVcc                    DQ           PVdq
%    _______    ____    ________    __________________    ___    ________________    _____________________    ________________    ____________________    ________________    _____________________    ________________    ____
%
%     0.99      1703      166       0.0974750440399295    red    471.596309196609    1.44012903404047e-104    294.473896975476    5.26938462847135e-66    766.070206172084    4.46651045074517e-167    31111.3682479554     0  
%
%
%    VaR_lvl    obs     num_hits        pct_fails        TL           LRuc                  PVuc                 LRcci                 PVcci                  LRcc                  PVcc                    DQ           PVdq
%    _______    ____    ________    _________________    ___    ________________    ____________________    ________________    ____________________    ________________    _____________________    ________________    ____
%
%     0.95      1703      284       0.166764533176747    red    311.998461569185    8.01452391622238e-70    197.952172065323    5.84419941129671e-45    509.950633634508    1.84343099243726e-111    9638.96137284591     0  
