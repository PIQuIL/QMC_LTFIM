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
#        # get an array of the possible hz values
#        tmp2 = dataN[np.where(dataN[:,1] == 0.5)]
#        tmp2 = tmp2[np.where(tmp2[:,2] == 0.0)]
#        
#        B = tmp1[:,2]
#        hz = tmp2[:,3]
#    
#        fig, ax = plt.subplots(2, 2, sharex=True)
#        for b in B:
#            filter1 = dataN[np.where(dataN[:,2] == b)]
#            for omega in hz:
#    
#                filter2 = filter1[np.where(filter1[:,3] == omega)]
#        
#                beta = filter2[:,1]
#                energy = filter2[:,4]
#    
#                ax[int(b*2)%2, floor(b)].plot(
#                    beta, 
#                    energy, 
#                    label=r'$\hz = {}$'.format(omega),
#                    ls='--'
#                )
#                ax[int(b*2)%2, floor(b)].set_title(r'$B = {}$'.format(b))
#               
#        ax[0,1].legend(loc=(1.05,0.5))
#        fig.text(0.5, 0.04, r'$\beta$', ha='center')
#        fig.text(0.04, 0.5, r'$\frac{E}{N}$', ha='center')
#        fig.savefig("N={}.pdf".format(N), dpi=1000, bbox_inches='tight')


def E_vs_N(N):

    # beta=2.BC_name=PBC_Dim=1_J=1_M=10000_hx=1_hz=1_skip=10_info.txt

    J = 1 # interaction strength

    # stuff for exact data 
    data_file = "energy_magnetization.txt"
    data = np.loadtxt(data_file, skiprows=1)
    dataN = data[np.where(data[:,0] == N)]

    # QMC energies
    main_path = "../data/sims/mixedstate/"
    QMC_files = os.listdir(main_path)
    for f in QMC_files:
        parsed = f.split('_')
        n = int(parsed[7].split('=')[1])
        if n != N or f.endswith('samples.txt') or f.endswith('state.jld2'):
            QMC_files.remove(f)

    beta_list = []
    hz_list = []
    hx_list = []
    for qmc_file in QMC_files:
        parsed = qmc_file.split('_')
      
        beta = float(parsed[0][5:(len(parsed[0])-3)])
        beta_list.append(beta)

        hz = float(parsed[6][3:len(parsed[6])])
        hz_list.append(hz)

        hx = float(parsed[5][3:len(parsed[5])])
        hx_list.append(hx)

    beta_list = np.unique(np.array(sorted(beta_list)))
    hz_list = np.unique(np.array(sorted(hz_list)))
    hx_list = np.unique(np.array(sorted(hx_list)))

    for idx, hx in enumerate(hx_list):
    
        num_rows = ceil(len(hz_list)/2)
        print(num_rows)
        fig, ax = plt.subplots(num_rows, 2, sharex=True)
        axs = [ax[i,j] for i in range(num_rows) for j in range(2)]

        filter1 = dataN[np.where(dataN[:,2] == hx)]
        for c, hz in enumerate(hz_list):

            filter2 = filter1[np.where(filter1[:,3] == hz)]
            beta = filter2[:,1]
            energy = filter2[:,4]

            axs[c].plot(beta, energy, label=r'Exact', color='orange')
            
            energies = np.zeros(len(beta_list))
            for i,beta in enumerate(beta_list):

                if beta%2 == 1 or beta%2 == 0:
                    beta = int(beta)

                if hz%2 == 1 or hz%2 == 0:
                    hz = int(hz)

                if hx%2 == 1 or hx%2 == 0:
                    hx = int(hx)

                # beta=2.BC_name=PBC_Dim=1_J=1_M=10000_hx=1_hz=0_nX=10_skip=10_info.txt
                #QMC_info_path = main_path + "beta={}.BC_name=OBC_Dim=1_J={}_hx={}_nX={}_skip=10_hz={}_info.txt".format(
                QMC_info_path = main_path + "beta={}.BC_name=PBC_Dim=1_J={}_M=10000_hx={}_hz={}_nX={}_skip=10_info.txt".format(
                    beta, 
                    J,
                    hx,
                    hz, 
                    N
                )

                QMC_info_file = open(QMC_info_path, "r").readlines()
                QMC_energy = QMC_info_file[3].split(' ')[4]

                energies[i] = QMC_energy

            axs[c].plot(beta_list, energies, ls='--', label=r'QMC', color='purple')
            axs[c].set_title(r'$h_z = {}$'.format(hz), fontsize=9)

        try:
            os.mkdir('hx={}/'.format(hx))
        except OSError:
            print("OSError: Could not make directory")

        axs[0].legend(loc='best', frameon=False)
        fig.text(0.5, 0.04, r'$\beta$', ha='center')
        fig.text(0.04, 0.5, r'$\frac{E}{N}$', ha='center')
        fig.savefig("hx={}/N={}_hx={}_EvsBeta_QMCvsExact.pdf".format(hx, N, hx), dpi=1000, bbox_inches='tight')

for N in [2, 4, 6, 8, 10]:
    E_vs_N(N)
