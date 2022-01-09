import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np

styles = ['solid', 'dashed', 'dotted', 'dashdot', 'dashed', 'dotted']
widths = [3, 3,3, 2, 2, 2]
cnt = 0
for f in glob.glob('*.csv'):
    if not '_NegBias' in f: continue
    __,molecule,___ = f.split('.')[0].split('_')
    lstyle = styles[cnt]
    lwidth = widths[cnt]
    cnt +=1

    data = pd.read_csv(f)
    for par in data.columns:
        if par == 'voltage': continue
        plt.figure(par)
        plt.title(par)
        plt.scatter(data['voltage'], data[par],zorder = cnt)
        plt.plot(data['voltage'], data[par], linestyle = lstyle, linewidth = lwidth, label=molecule, zorder = cnt)
        plt.legend()
        plt.tight_layout()
        plt.savefig('%s_NegBias.png' % par)


