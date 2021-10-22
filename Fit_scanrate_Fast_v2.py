import pandas as pd
import numpy as np
import funcs as fun
import matplotlib.pyplot as plt
import scipy.optimize as sco
import time
start = time.time()

Fixed = {
    # 'n'       : 300,
    # 'gammaL'  : 0.0142,
    # 'gammaR'  : 0.0256,
    # 'kappa'   : 10.1,
    'sigma'   : 0,
    # 'E_AB'    : 0.82,
    # 'E_AC'    :-0.301,
    # 'chi'     : 1.59,
    # 'eta'     : 0.67,
    # 'gam'     : 20,
    'lam'     : 1.2,
    'P'       : 0,
    'u'       : 10/1000
}
Fit = True

initpar = {
    'n':	500000,
 	  'gammaR':	4e-06,
 	  'gammaL':	0.00139,
 	  'kappa':	10.1,
 	  'sigma':	0,
      'E_AB':	0.81,
      'E_AC':	-1.14,
 	  'chi':	0.72,
 	  'eta':	0.67,
 	  'gam':	20,
 	  'lam':	2,
    'P'       : 0,
    'u'       : 10/1000
}
a = .75
bnds = {
        'n'       : [100,10E10],
        'gammaL'  : [1E-4,.1],
        'gammaR'  : [1E-4,.1],
        'kappa'   : [1E-6,10],
        'sigma'   : [1E-6,10],
        'E_AB'    : [0,1.5],
        'E_AC'    : [-1.5,1.5],
        'chi'     : [0,2.0],
        'eta'     : [0,1],
        'gam'     : [1E-6,200],
        'lam'     : [1E-6,2],
        'P'       : [0,1],
        'u'       : [0,300]
    }

for key in Fixed.keys():
        del bnds[key]
paramsCopy = initpar.copy()     
for key in Fixed.keys():
    initpar[key] = Fixed[key]
    del paramsCopy[key]
    
def fitfunc(x):
    params = initpar.copy()
    for i,key in enumerate(list(paramsCopy.keys())):
        params[key] = x[i]
    
    #%% Setting up the voltage
    CurrDF = pd.DataFrame()
    CurrDF['Voltage'] = np.round(np.arange(-2,2,0.01),2)
    
    #%% Calculating the Currents
    eqI = np.vectorize(fun.reducedLandauer)
    
    CurrDF['I_np'] = params['n']*eqI(CurrDF['Voltage'], params['gammaL'], params['gammaR'], params['E_AB'], params['eta'], params['sigma'])
    CurrDF['I_p'] = params['n']*eqI(CurrDF['Voltage'], params['gammaL']*params['kappa'], params['gammaR']*params['kappa'], params['E_AB']+params['chi'], params['eta'], params['sigma'])
    
    # %% Calculating the Average Bridge Populations
    eqN = np.vectorize(fun.reducedBridgePop)
    
    CurrDF['n_np'] = eqN(CurrDF['Voltage'], params['gammaL'], params['gammaR'], params['E_AB'], params['eta'])
    CurrDF['n_p'] = eqN(CurrDF['Voltage'], params['gammaL']*params['kappa'], params['gammaR']*params['kappa'], params['E_AB']+params['chi'], params['eta'])
    
    #%% Calculating the Marcus Rates
    eqR = np.vectorize(fun.MarcusETRates)
    
    CurrDF['R_AC'], CurrDF['R_CA'] = eqR(CurrDF['Voltage'], params['gam'], params['lam'], params['E_AC'])
    CurrDF['R_BD'], CurrDF['R_DB'] = eqR(CurrDF['Voltage'], params['gam']*params['kappa'], params['lam'], params['E_AC']+params['chi'])
    
    #%% Calculating Ks
    CurrDF['k_S0_S1'] = (1-CurrDF['n_np'])*CurrDF['R_AC'] + CurrDF['n_np']*CurrDF['R_BD']
    CurrDF['k_S1_S0'] = (1-CurrDF['n_p'])*CurrDF['R_CA'] + CurrDF['n_p']*CurrDF['R_DB']
    
    #%% Plotting
    # plt.figure('Raw1')
    # plt.plot(CurrDF['Voltage'],CurrDF['k_S0_S1'])
    # plt.plot(CurrDF['Voltage'],CurrDF['k_S1_S0'])
    
    # %% Calcuating Probability and Current
    diff = 0
    ran = [10,14,17,20,25,36,100,140,166,200,250]
    ran = [10]
    # ran = [250,200,166,140,100,36,25,20,17,14,10]
    P = params['P']
    data = pd.read_csv('Data\\SRt_cont_Normalized.txt', delimiter = '\t')
    for colV in data.columns:   
        if colV[:-2][-1] == 'C':
                continue
        if not float(colV[:-3]) in ran:
            continue
        
        colC = colV.replace('V','C')
        colthr = colV.replace('V','thr')
        colP   = colV.replace('V','P')
        
        subset = pd.DataFrame()
        subset[colV] = data[colV]
        subset[colC] = data[colC]
        subset = subset.dropna()
        
        val = int(colV[0:3])
        
        delt = abs(data[colV][2]-data[colV][3])/(val/1000)
        I = []
        Parray = []
        delArray = []
        for i,v in enumerate(subset[colV]):
            tempDf =CurrDF[CurrDF['Voltage']==np.round(v,2)].reset_index()
            calcs = dict(tempDf.iloc[0])
            Parray += [P]
            I += [((1-P)*calcs['I_np']+P*calcs['I_p'])]
            
            dPdt = calcs['k_S0_S1']-P*(calcs['k_S0_S1']+calcs['k_S1_S0'])
            delArray += [dPdt]
            P = P+dPdt*delt
        subset[colthr] = pd.Series(I)
        diff += np.sum(np.subtract(subset[colC],subset[colthr])**2)/(np.max(subset[colC])-np.min(subset[colC]))
        if not Fit:            
            plt.figure('Current')
            plt.plot(subset[colV],subset[colC], color = 'black')
            plt.plot(subset[colV],subset[colthr], label = colthr)
            plt.legend()
            
            plt.figure('Probability')
            plt.plot(subset[colV],Parray, label = colthr)
            plt.legend()
    fun.saveParams(params,np.sqrt(diff),'Params\\All\\SR_tot.txt')
    if Fit:
        print(np.log10(np.sqrt(diff)))
    return np.sqrt(diff)

if not Fit:
    print(fitfunc(list(paramsCopy.values())))
else:
    # result = sco.minimize(fitfunc,x0 = list(paramsCopy.values()), bounds = list(bnds.values()))
    result = sco.differential_evolution(fitfunc,bounds = list(bnds.values()))
    print('Total Time: %d'%(time.time()-start)) 
    
    for i,key in enumerate(list(paramsCopy.keys())):
        initpar[key] = result.x[i]
        paramsCopy[key] = result.x[i]
    
    # diff = result.fun
    # fun.saveParams(initpar,diff,'Params\\paramsSRReverseTot.txt')
    # fun.saveParams(initpar,diff,'Params\\All\\paramsSRReverseTot.txt')