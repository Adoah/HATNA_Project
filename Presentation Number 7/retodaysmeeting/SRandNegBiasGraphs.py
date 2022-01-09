import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np

scatterDataName = 'Data_ScatterPlots.csv'
errorMinName = 'Data_ErrorPlotsMin.csv'
errorMaxName = 'Data_ErrorPlotsMax.csv'

scatterData = pd.read_csv(scatterDataName,index_col='DatSet')
errorMinData = pd.read_csv(errorMinName,index_col='DatSet')
errorMaxData = pd.read_csv(errorMaxName,index_col='DatSet')

errorList = [set for set in scatterData.index if 'Neg' in set]
SRList = [set for set in scatterData.index if not 'Neg' in set]

splitNames = [i.split('_')[0] for i in scatterData.index]
splitErrorList = [i.split('_')[0] for i in errorList]

NegAverages = scatterData.loc[errorList].copy()

for par in scatterData.columns:
    plt.figure(par)
    plt.title(par)
    plt.scatter(splitNames,scatterData[par],color = 'red', label = 'Fit Scan Rate', zorder = 5)
    plt.scatter(splitErrorList,NegAverages[par], color='blue',zorder = 6)
    minsBar = NegAverages[par]-errorMinData[par]
    maxsBar = errorMaxData[par]- NegAverages[par]
    plt.errorbar(splitErrorList,NegAverages[par], [minsBar,maxsBar], fmt = '.',color = 'blue', elinewidth=3,label = 'Fit Negative Bias', zorder = 3)

    if par == 'chi':
        UVMeas = [2.79, 2.74, 2.60, 2.28, 2.74, 2.82]
        plt.scatter(splitErrorList, UVMeas, color='black', label='UV measurements',zorder = 7)

    plt.xticks(rotation = 45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('%s.png'%par)



# summary = pd.read_csv('Summary of Params.csv', index_col='DatSet')
# print(summary)
# plt.figure('chi')
# plt.scatter(summary.index,summary['chi'])
#
# plt.show()


