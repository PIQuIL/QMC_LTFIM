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


def E_vs_Ns():

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


def E_vs_N():

    data_file = "../energy_magnetization.txt"
    data = np.loadtxt(data_file, skiprows=1)
    
    markers = ['v', '^', '<', '>']
    
    N = 4
    J = 1 # interaction strength
    h = 1 # transverse field strength

    dataN = data[np.where(data[:,0] == N)]
    
    # get an array of the possible Omega values
    tmp2 = dataN[np.where(dataN[:,1] == 0.5)]
    tmp2 = tmp2[np.where(tmp2[:,2] == 0.0)]
    
    Omega = tmp2[:,3]
    
    fig, ax = plt.subplots(2, 2, sharex=True)
    #ax4 = plt.subplot(2, 2, 4)
    #ax3 = plt.subplot(2, 2, 3)
    #ax2 = plt.subplot(2, 2, 2, sharex=ax4)
    #ax1 = plt.subplot(2, 2, 1, sharex=ax3)

    axs = [ax[0,0], ax[0,1], ax[1,0], ax[1,1]]

    filter1 = dataN[np.where(dataN[:,2] == h)]

    #QMC_energies = np.array([ 
    #    [-0.7245200000000001, -1.0194800000000002, -1.1093250000000001, -1.1414387499999998, -1.1594080000000004, -1.1673524999999998, -1.1713235714285717, -1.1788912499999999, -1.1820855555555556, -1.1847200000000000],
    #    [-7.0219199999999997, -7.0020349999999993, -6.9906483333333345, -6.9949250000000003, -6.9899780000000025, -6.9926283333333341, -6.9875742857142864, -6.9873531250000003, -6.9874494444444419, -6.9852895000000004],
    #    [-13.0193550000000009, -13.0007850000000005, -12.9885249999999992, -12.9963525000000004, -12.9837239999999987, -12.9971108333333341, -12.9877471428571418, -12.9904712499999988, -12.9909005555555552, -12.9886035000000000],
    #    [-18.9985099999999996, -18.9882874999999984, -18.9745850000000011, -18.9936112500000007, -18.9917760000000015, -18.9956516666666708, -18.9935314285714263, -18.9835443750000010, -18.9870427777777806, -18.9891079999999981]
    #])

    for omega in Omega:

        idx = int(omega)
        filter2 = filter1[np.where(filter1[:,3] == omega)]
    
        beta = filter2[:,1]
        energy = filter2[:,4]

        QMC_energies = np.zeros_like(energy)

        main_path = "../data/sims/mixedstate/"
        for i, b in enumerate(np.arange(0.5, 5.5, 0.5)):
       
            if b % 2 == 1 or b % 2 == 0:
                b = int(b)

            QMC_info_path = main_path + "beta={}.BC_name=OBC_Dim=1_J={}_h={}_nX=4_skip=10_Ω={}_info.txt".format(
                b, 
                J, 
                h, 
                int(omega)
            )

            QMC_info_file = open(QMC_info_path, "r").readlines()
            QMC_energy = QMC_info_file[3].split(' ')[4]
            QMC_energies[i] = QMC_energy
  
        ax_ = axs[idx]

        ax_.plot(
            beta, 
            energy, 
            ls='--',
            label='Exact'
        )

        ax_.plot(
            beta, 
            QMC_energies,
            label='QMC'
        )
        ax_.set_title(r'$\Omega = {}$'.format(omega), fontsize=10)
           
    ax_.legend(frameon=False)
    fig.text(0.5, 0.04, r'$\beta$', ha='center')
    fig.text(0.04, 0.5, r'$\frac{E}{N}$', ha='center')
    fig.savefig("N={}_QMCvsExact.pdf".format(N), dpi=1000, bbox_inches='tight')

E_vs_N()
