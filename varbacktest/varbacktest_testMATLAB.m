% MathWorks, (2012). Risk Management Toolbox: User's Guide (R2021b).

format long 

% Data collection
URL = ["https://git.io/J1clf"];
filename = "toyserie.xlsx";
urlwrite(URL, filename);
df = readtimetable("toyserie.xlsx");

% specifications
VaR_ops = ["VaR_1", "VaR_5"];
conf_lvl_ops = [0.99, 0.95];
returns = df.mean_true;

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
    
    results = table(conf_lvl_ops(i), obs, num_hits, pct_fails, TL, LRuc, PVuc, LRcci, ...
                    PVcci, LRcc, PVcc, 'VariableNames', {'VaR_lvl', ...
                    'obs', 'num_hits', 'pct_fails', 'TL', 'LRuc', 'PVuc', 'LRcci', 'PVcci', ...
                    'LRcc', 'PVcc'});
    disp(results)
end

%     VaR_lvl    obs     num_hits        pct_fails        TL           LRuc          PVuc         LRcci                  PVcci                  LRcc          PVcc
%    _______    ____    ________    _________________    ___    ________________    ____    ________________    _____________________    ________________    ____
%
%     0.99      1703      1278      0.750440399295361    red    9865.69395900987     0      890.478433173468    1.15267565481073e-195    10756.1723921833     0  
%
%
%    VaR_lvl    obs     num_hits        pct_fails        TL           LRuc          PVuc         LRcci                  PVcci                  LRcc          PVcc
%    _______    ____    ________    _________________    ___    ________________    ____    ________________    _____________________    ________________    ____
%
%     0.95      1703      999       0.586611861421022    red    3748.19441993176     0      862.079510341855    1.71979596437219e-189    4610.27393027362     0  
