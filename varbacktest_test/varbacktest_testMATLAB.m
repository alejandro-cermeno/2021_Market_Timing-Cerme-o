format long 

% Data collection
URL = ["https://git.io/J1Eqb"];
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
    VaR = df.(i + 1); % VaR serie

    % object creation
    vbt = varbacktest(returns, VaR, 'VaRLevel', conf_lvl_ops(i));
    tlObj = tl(vbt);
    ucObj = pof(vbt);
    cciObj = cci(vbt);
    ccObj = cc(vbt);

    % some metrics
    obs = ucObj.Observations;
    num_hits = ucObj.Failures;
    pct_fails = num_hits/obs;

    % Engle and Manganelli (2004)
    [DQ, PVdq] = dq(returns, VaR, conf_lvl_ops(i));
    
    % results table
    results = table(conf_lvl_ops(i), obs, num_hits, pct_fails, tlObj.TL,...
                    ucObj.LRatioPOF, ucObj.PValuePOF, ucObj.POF, ...
                    cciObj.LRatioCCI, cciObj.PValueCCI, cciObj.CCI, ...
                    ccObj.LRatioCC, ccObj.PValueCC, ccObj.CC, DQ, ...
                    PVdq, 'VariableNames', {'VaR_lvl', 'obs', ...
                    'num_hits', 'pct_fails', 'TL', 'LRuc', 'PVuc', ...
                    'UC', 'LRcci', 'PVcci', 'CCI', 'LRcc', 'PVcc', ...
                    'CC', 'DQ', 'PVdq'});
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

%    VaR_lvl    obs     num_hits        pct_fails         TL           LRuc                  PVuc               UC           LRcci                 PVcci             CCI            LRcc                  PVcc               CC             DQ           PVdq
%    _______    ____    ________    __________________    ___    ________________    _____________________    ______    ________________    ____________________    ______    ________________    _____________________    ______    ________________    ____
%
%     0.99      1703      166       0.0974750440399295    red    471.596309196609    1.44012903404047e-104    reject    294.473896975476    5.26938462847135e-66    reject    766.070206172084    4.46651045074517e-167    reject    31111.3682479554     0  
%
%    VaR_lvl    obs     num_hits        pct_fails        TL           LRuc                  PVuc              UC           LRcci                 PVcci             CCI            LRcc                  PVcc               CC             DQ           PVdq
%    _______    ____    ________    _________________    ___    ________________    ____________________    ______    ________________    ____________________    ______    ________________    _____________________    ______    ________________    ____
%
%     0.95      1703      284       0.166764533176747    red    311.998461569185    8.01452391622238e-70    reject    197.952172065323    5.84419941129671e-45    reject    509.950633634508    1.84343099243726e-111    reject    9638.96137284591     0  
