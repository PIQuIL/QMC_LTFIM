import matplotlib.pyplot as plt
import numpy as np

from math import floor

params = {'text.usetex': True,                                                   
            'font.family': 'serif',                                              
            'legend.fontsize': 10,                                               
            'axes.labelsize': 10,                                                
            'xtick.labelsize':10,                                                
            'ytick.labelsize':10,                                                
            'lines.linewidth':1,                                                 
            "patch.edgecolor": "black"                                           
         }                                                                       
                                                                                 
plt.rcParams.update(params)                                                      
plt.style.use('seaborn-deep') 

data_file = "energy_magnetization.txt"
data = np.loadtxt(data_file, skiprows=1)

markers = ['v', '^', '<', '>']

for N in range(2,11):

    dataN = data[np.where(data[:,0] == N)]
   
    # get an array of the possible B values
    tmp1 = dataN[np.where(dataN[:,1] == 0.5)]
    tmp1 = tmp1[np.where(tmp1[:,3] == 0.0)]
    # get an array of the possible Omega values
    tmp2 = dataN[np.where(dataN[:,1] == 0.5)]
    tmp2 = tmp2[np.where(tmp2[:,2] == 0.0)]
    
    B = tmp1[:,2]
    Omega = tmp2[:,3]

    fig, ax = plt.subplots(2, 2, sharex=True)
    for b in B:
        filter1 = dataN[np.where(dataN[:,2] == b)]
        for omega in Omega:

            filter2 = filter1[np.where(filter1[:,3] == omega)]
    
            beta = filter2[:,1]
            energy = filter2[:,4]

            ax[int(b)%2, floor(int(b / 2))].plot(
                beta, 
                energy, 
                label=r'$\Omega = {}$'.format(omega),
                ls='--'
            )
            ax[int(b)%2, floor(int(b / 2))].set_title(r'$B = {}$'.format(b))
           
    ax[0,1].legend(loc=(1.05,0.5))
    fig.text(0.5, 0.04, r'$\beta$', ha='center')
    fig.text(0.04, 0.5, r'$\frac{E}{N}$', ha='center')
    fig.savefig("N={}.pdf".format(N), dpi=1000, bbox_inches='tight')
