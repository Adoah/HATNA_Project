import pandas as pd
import numpy as np
import funcs as fun
import matplotlib.pyplot as plt
import scipy.optimize as sco
import time
start = time.time()


Fit = False 
Fixed ={ 
         
        'n'	    :		        1.200e+06,
    	'gammaL'    :			6.53e-06,
   	'gammaR'    :			2.30e-03,
    	'kappa'     :			13.0,
 #   	'sigma'     :			5.09e-02,
   	'E_AB'      :			7.50e-01,
   	'E_AC'      :			-.800e+00,
   	'chi'       :			9.00e-01,
   	'eta'       :			6.80e-01,
 #  	'gam'       :			5.50e+00,
   	'lam'       :			1.20e+00,
 #	'P'         :			0.12, 
	'u'         :			14
        
}

bnds = {
        'n'       : [1e03,5e07], 
        'gammaL'  : [7e-08,1.3e-05],
        'gammaR'  : [7e-08, 0.0046],
        'kappa'   : [1,16],
        'sigma'   : [0.0906,.3], 
        'E_AB'    : [0.2,1.06],
        'E_AC'    : [-1.610, -0.2], 
        'chi'     : [0.47, 1.90],
        'eta'     : [0, 1],
        'gam'     : [1,20],
        'lam'     : [0.5, 1.8], 
        'P'       : [0,1], 
        'u'       : [0,300]
    }					

initpar ={ 
		  
	
	'n'          :			8.65e+04,
	'gammaL'     :			6.53e-06,
	'gammaR'     :			2.30e-03,
	'kappa'      :			1.30e+01,
	'sigma'      :			0.00e+00,
	'E_AB'       :			7.50e-01,
	'E_AC'       :			-1.00e+00,
	'chi'        :			9.00e-01,
	'eta'        :			7.00e-01,
	'gam'        :			5.50e+00,
	'lam'        :			1.20e+00,
	'P'          :			0,
	'u'          :			14.00e-03

       
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
    
    CurrDF['I_np'] = eqI(CurrDF['Voltage'],params['n'],params['gammaL'], params['gammaR'], params['E_AB'], params['eta'], params['sigma'])
    CurrDF['I_p'] = eqI(CurrDF['Voltage'],params['n'],params['gammaL']*params['kappa'], params['gammaR']*params['kappa'], params['E_AB']+params['chi'], params['eta'], params['sigma'])
    
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
    ran = [10,14,17,20,25,36,50,100,140,166,200,250]
    ran = [14]  
    #ran = [250,200,166,140,100,50,36,25,20,17,14,10]
    P = params['P'] 
    
    data = pd.read_csv('Data/F4_SR.txt', delimiter = '\t')   
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
            P = P +dPdt*delt
        subset[colthr] = pd.Series(I)
       

        
        diff += np.sum(np.subtract(subset[colC],subset[colthr])**2)/(np.max(subset[colC])-np.min(subset[colC]))
        if not Fit:            
            #plt.figure('Current')
            #plt.plot(subset[colV],subset[colC], color = 'black')
            #plt.plot(subset[colV],subset[colthr], label = colthr)
            #plt.legend()
            
            plt.figure('Probability')
            plt.plot(subset[colV],Parray, label = colthr)
            plt.legend()
            plt.show()
            
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
       
