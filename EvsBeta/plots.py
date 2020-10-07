import matplotlib.pyplot as plt
import numpy as np
import os

from math import floor, ceil

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

#def E_vs_Ns():
#
#    data_file = "../energy_magnetization.txt"
#    data = np.loadtxt(data_file, skiprows=1)
#    
#    markers = ['v', '^', '<', '>']
#
#    J = 1 # interaction strength
#    
#    for N in np.arange(4,12,2):
#    
#        dataN = data[np.where(data[:,0] == N)]
#       
#        # get an array of the possible B values
#        tmp1 = dataN[np.where(dataN[:,1] == 0.5)]
#        tmp1 = tmp1[np.where(tmp1[:,3] == 0.0)]
#        # get an array of the possible Omega values
#        tmp2 = dataN[np.where(dataN[:,1] == 0.5)]
#        tmp2 = tmp2[np.where(tmp2[:,2] == 0.0)]
#        
#        B = tmp1[:,2]
#        Omega = tmp2[:,3]
#    
#        fig, ax = plt.subplots(2, 2, sharex=True)
#        for b in B:
#            filter1 = dataN[np.where(dataN[:,2] == b)]
#            for omega in Omega:
#    
#                filter2 = filter1[np.where(filter1[:,3] == omega)]
#        
#                beta = filter2[:,1]
#                energy = filter2[:,4]
#    
#                ax[int(b*2)%2, floor(b)].plot(
#                    beta, 
#                    energy, 
#                    label=r'$\Omega = {}$'.format(omega),
#                    ls='--'
#                )
#                ax[int(b*2)%2, floor(b)].set_title(r'$B = {}$'.format(b))
#               
#        ax[0,1].legend(loc=(1.05,0.5))
#        fig.text(0.5, 0.04, r'$\beta$', ha='center')
#        fig.text(0.04, 0.5, r'$\frac{E}{N}$', ha='center')
#        fig.savefig("N={}.pdf".format(N), dpi=1000, bbox_inches='tight')


def E_vs_N(N):

    J = 1 # interaction strength

    # stuff for exact data 
    data_file = "../energy_magnetization.txt"
    data = np.loadtxt(data_file, skiprows=1)
    dataN = data[np.where(data[:,0] == N)]

    # QMC energies
    main_path = "../data/sims/mixedstate/"
    QMC_files = os.listdir(main_path)
    for f in QMC_files:
        parsed = f.split('_')
        n = int(parsed[5].split('=')[1])
        if n != N or f.endswith("samples.txt"):
            QMC_files.remove(f)

    beta_list = []
    Omega_list = []
    h_list = []
    for qmc_file in QMC_files:
        parsed = qmc_file.split('_')
        
        beta = float(parsed[0][5:(len(parsed[0])-3)])
        beta_list.append(beta)

        Omega = float(parsed[7][2:len(parsed[7])])
        Omega_list.append(Omega)

        h = float(parsed[4][2:len(parsed[4])])
        h_list.append(h)

    beta_list = np.unique(np.array(sorted(beta_list)))
    Omega_list = np.unique(np.array(sorted(Omega_list)))
    h_list = np.unique(np.array(sorted(h_list)))

    for idx,h in enumerate(h_list):
    
        num_rows = ceil(len(Omega_list)/2)
        fig, ax = plt.subplots(num_rows, 2, sharex=True)
        axs = [ax[i,j] for i in range(num_rows) for j in range(2)]

        filter1 = dataN[np.where(dataN[:,2] == h)]
        for c, omega in enumerate(Omega_list):

            filter2 = filter1[np.where(filter1[:,3] == omega)]
            beta = filter2[:,1]
            energy = filter2[:,4]

            axs[c].plot(beta, energy, label=r'Exact', color='orange')
            
            energies = np.zeros(len(beta_list))
            for i,beta in enumerate(beta_list):

                if beta%2 == 1 or beta%2 == 0:
                    beta = int(beta)

                if omega%2 == 1 or omega%2 == 0:
                    omega = int(omega)

                if h%2 == 1 or h%2 == 0:
                    h = int(h)

                QMC_info_path = main_path + "beta={}.BC_name=OBC_Dim=1_J={}_h={}_nX={}_skip=10_Ω={}_info.txt".format(
                    beta, 
                    J,
                    h,
                    N, 
                    omega
                )

                QMC_info_file = open(QMC_info_path, "r").readlines()
                QMC_energy = QMC_info_file[3].split(' ')[4]

                energies[i] = QMC_energy

            axs[c].plot(beta_list, energies, ls='--', label=r'QMC', color='purple')
            axs[c].set_title(r'$\Omega = {}$'.format(omega), fontsize=9)

        try:
            os.mkdir('h={}/'.format(h))
        except OSError:
            print("OSError: Could not make directory")

        axs[0].legend(loc='best', frameon=False)
        fig.text(0.5, 0.04, r'$\beta$', ha='center')
        fig.text(0.04, 0.5, r'$\frac{E}{N}$', ha='center')
        fig.savefig("h={}/N={}_h={}_EvsBeta_QMCvsExact.pdf".format(h, N, h), dpi=1000, bbox_inches='tight')

for N in [4,6,8]:
    E_vs_N(N)
