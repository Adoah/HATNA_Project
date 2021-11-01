import glob
from random import choice
import pandas as pd
import time
from funcs import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as sco
from scipy.stats import linregress
start = time.time()
colors = ['gray', 'red', 'peru', 'orangered', 'orange', 'darkgoldenrod', 'yellow', 'darkolivegreen', 'forestgreen', 'darkgreen', 'lime', 'green', 'seagreen', 'springgreen', 'turquoise', 'teal', 'aqua', 'dodgerblue', 'navy', 'blue', 'fuchsia', 'deeppink', 'crimson']
Fixed = {
    # 'n'       : 300,
    # 'gammaL'  : 0.0142,
    # 'gammaR'  : 0.0256,
    # 'kappa'   : 0.261,
    # 'sigma'   : 0,
    # 'E_AB'    : 0.775,
    # 'E_AC'    :-0.301,
    # 'chi'     : 1.59,
    # 'eta'     : 0.625,
    # 'gam'     : 20,
    # 'lam'     : 2.09,
    'P'       : 0,
    'u'       : 10/1000
}
bnds = {
        'n'       : [1,1000],
        'gammaL'  : [1E-6, 0.1],
        'gammaR'  : [1E-6, 0.1],
        'kappa'   : [1E-6,10],
        'sigma'   : [1E-6,.5],
        'E_AB'    : [0.5,1.5],
        'E_AC'    : [-.75,1],
        'chi'     : [0,2],
        'eta'     : [0,1],
        'gam'     : [1E-2, 300],
        'lam'     : [0.5, 3],
        'P'       : [0,1],
        'u'       : [0,300]
    }
ax =.90
key2 = 'P'
c = 'crimson'
print(c)
a =11

params = {
    'n'       : 3.0029e+02,
    'gammaL'  : 1.2366e-02,
    'gammaR'  : 2.5545e-02,
    'kappa'   : 2.3379e-01,
    'sigma'   : 7.6073e-02,
    'E_AB'    : 9.7558e-01,
    'E_AC'    :-3.1251e-01,
    'chi'     : 1.4667e+00,
    'eta'     : 7.6600e-01,
    'gam'     : 1.9967e+01,
    'lam'     : 2.0745e+00,
    'P'       : 0,
    'u'       : 10/1000
}
for key in Fixed.keys():
        del bnds[key]

# %% Making sure that the params are within the bounds
# for key in bnds.keys():
#     if bnds[key][0] > params[key] or bnds[key][1] < params[key]:
#         print(key)
#         params[key] = np.mean(bnds[key])

# %% Making a copy of the Parameters with the 'fixed' ones removed
paramsCopy = params.copy()     
for key in Fixed.keys():
    params[key] = Fixed[key]
    del paramsCopy[key]

ForOrigin = pd.DataFrame()

def fitfunc(x):
    temp = params.copy()
    for i,key in enumerate(list(paramsCopy.keys())):
        temp[key] = x[i]
    diff = 0
    diff1 = 0
    Pf = temp['P']
    exp_array = []
    thr_array = []
    exp_vndr = []
    thr_vndr = []
    
    ran = [250,200,166,140,100,36,25,20,17,14,10]
    ran = [10,14,17,20,25,36,100,140,166,200,250]
    # ran = [10,14,17,20,25,36]
    # ran = [10,14]
    for val in ran:
        temp['u'] = val/1000
        temp['P'] = Pf
        # print(temp['u'])
            
        nm = 'Data\\Air_Rate%03.d.txt'%val
        data = pd.read_csv(nm, delimiter = '\t',skiprows =1, header = None)
        
        XMod, YMod, P = NitzFit(list(data[3]), *list(temp.values()))
        data[3] = np.round(data[3]/2,2)*2
        data[5] = YMod
        Pf = P[-1]
        data[4] = data[4]*1E9
        data[5] = data[5]*1E9
        data[6] = P
        
        # ForOrigin['X_%d'%val] = data[3]
        # ForOrigin['P_%d'%val] = data[6]
        # ForOrigin['Y_exp_%d'%val] = data[4]
        # ForOrigin['Y_thr_%d'%val] = data[5]
        
        diff += np.sum(np.subtract(data[4],data[5])**2)/(np.max(data[4])-np.min(data[4]))
        
        #%% Grabbing the Peak Difference
        minLoc_exp = data[4].idxmin()
        minLoc_thr = data[5].idxmin()

        df_exp = data[data[3] == data.iloc[minLoc_exp][3]]
        df_thr = data[data[3] == data.iloc[minLoc_thr][3]]
        
        exp = df_exp.iloc[1][4]-df_exp.iloc[0][4]
        if len(df_thr) == 2:
            thr = df_thr.iloc[1][5]-df_thr.iloc[0][5]
        else:
            thr = 0
        diff1 += (exp-thr)**2
        
        exp_array += [exp]
        thr_array += [thr]
        
        exp_vndr +=[data.iloc[minLoc_exp][3]]
        thr_vndr +=[data.iloc[minLoc_thr][3]]
        
        # if val == 14:
        # plt.figure('P3')
        # plt.plot(data[3],P,label = val)
        # plt.legend()
        # plt.figure('I3')
        # plt.plot(data[3],data[4], color = 'black')
        # plt.plot(data[3],data[5])#,label = val)
        # plt.legend()
        # # plt.scatter(df_thr.iloc[0][3],df_thr.iloc[0][5])
        # # #     # plt.legend()
            
    saveParams(temp,np.sqrt(diff),'Plots\\paramsSRReverseJust10-36.txt')
    
    # # diff1 = 2*diff1/(np.max(exp_array)-np.min(exp_array))
    # # diff = diff/15+diff1
    # # saveParams(temp,np.sqrt(diff),'Params\\paramsSlope3_7.txt')
    # # print('%.3f'%np.sqrt(np.sqrt(diff1)))
    
    # slope_exp = linregress(ran[1:],exp_array[1:]).slope
    # slope_thr = linregress(ran[1:],thr_array[1:]).slope

    # diff2 = (slope_exp-slope_thr)**2
    
    ForOrigin['Rate'] = ran
    # ForOrigin['Exp']  = exp_array
    # ForOrigin['Thy']  = thr_array
    ForOrigin['Exp']  = exp_vndr
    ForOrigin['Thy']  = thr_vndr
    print(val)
    plt.figure('Peak Diff')
    plt.scatter(ran,exp_array,color = 'black')
    plt.plot(ran,exp_array,color = 'black')
    plt.scatter(ran,thr_array,color = c)
    plt.plot(ran,thr_array,color = c)
    
    # eq1 = np.vectorize(reducedLandauer)
    
    # I1 = eq1(data[3],temp['gammaL'], temp['gammaR'], temp['E_AB'], temp['eta'], temp['sigma'])
    # I2 = eq1(data[3],temp['gammaL']*temp['kappa'], temp['gammaR']*temp['kappa'], temp['E_AB']+temp['chi'], temp['eta'], temp['sigma'])
    
    # I1 = temp['n']*I1*1E9
    # I2 = temp['n']*I2*1E9
   
    # plt.figure('I3')
    # plt.plot(data[3],I1, color = 'red', label = 'HC/LE')
    # plt.plot(data[3],I2, color = 'blue', label = 'LC/HE')
    # plt.legend()
    # # print(diff2/10)
    # # print(diff)
    # # diffR = diff2/10+diff
    print('%.3f'%np.sqrt(diff))
    return np.sqrt(diff)


# V = paramsCopy[key2]
# for a in sorted(np.append([1],np.linspace(ax,1/ax,6))):
#     paramsCopy[key2]=V*a
#     test = fitfunc(list(paramsCopy.values()))
#     print('%.2f\t%.2f'%(a,np.sqrt(test)))

print(fitfunc(list(paramsCopy.values())))


# result = sco.minimize(fitfunc,x0 = list(paramsCopy.values()), bounds = list(bnds.values()))
# print(result)
# result = sco.differential_evolution(fitfunc,bounds = list(bnds.values()))

print('Total Time: %d'%(time.time()-start))  
# for i,key in enumerate(list(paramsCopy.keys())):
#     params[key] = result.x[i]
#     paramsCopy[key] = result.x[i]

# diff = result.fun
# saveParams(params,diff,'Params\\paramsSRReverseJust10-36.txt')
# saveParams(params,diff,'Params\\All\\paramsSRReverseJust10-36.txt')