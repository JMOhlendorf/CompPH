"""
Python version 3.7.7

@author: Chritian Gorjaew, Julius Meyer-Ohlendorf
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from tqdm import tqdm

@njit  # gives significant speedup by precompiling the following function
def MMC_step(lattice, Energy, beta):
    '''
    Performs one step of the MMC algorithm on the lattice. One MMC step in this
    context are N lattice updates at random positions.

    :param lattice: lattice used to perform one MMC step
    :type lattice: ndarry shape (N+2,), shape due to free BC
    :param Energy: Energy of the system in the given lattice configuration
    :type Energy: float
    :param beta: inverse temperature
    :type beta: : float

    :returns: Energy of the lattice configuration after N updates
    :rtype: ndarray
    '''
    
    N = lattice.shape[0] - 2
    
    for k in range(N):
        
        ridx = np.random.randint(1, N+1)
        idx_l = ridx - 1
        idx_r = ridx + 1

        delta_E = 2 * lattice[ridx] * (lattice[idx_l] + lattice[idx_r])
        q = np.exp(- beta * delta_E)
        r = np.random.uniform(0., 1.)
                
        if q > r:
            lattice[ridx] *= -1
            Energy += delta_E
    
    return Energy


'''
############## Performing simuation for desired parameters ##############
'''
simulation = True
evaluation = True
path = "./"
      
T = np.arange(0.2, 4.2, 0.2)
beta = 1. / T

N_spin = np.array([10, 100, 1000])
N_samples = np.array([1000, 10000])

N_wait_cold = 10000  # temperature dependent number of thermalization steps
N_wait_warm = 1000

if simulation:
    
    for n in tqdm(N_spin):
        
        for N in N_samples:
            
            for i in range(beta.shape[0]):
                b = beta[i]
                lattice = np.random.choice([-1,1], size=(n + 2,))
                lattice[0], lattice[-1] = 0, 0
                E_wait = -np.sum(lattice[1:-2] * lattice[2:-1])
                
                N_wait = N_wait_warm if T[i] >= 1.6 else N_wait_cold
                
                # thermalization
                for j in range(N_wait):
                    E_wait = MMC_step(lattice, E_wait, b)
                    
                E = np.zeros(N + 1)  # here the energies of the production run 
                                     # are stored
                E[0] = E_wait

                for k in range(N):
                    E[k+1] = MMC_step(lattice, E[k], b)
                    
                # energies are stored and further analyzed below
                save_path = path + "N{0}_Nsample{1}_T{2:.1f}.npz".format(n, N, 
                                                                         T[i])
                np.savez(save_path, Energy = E)
      
'''
############ Evaluation and plotting #############
'''
if evaluation:
        
    # Arrays storing qunatites for different temperatures, system and samples
    # sizes
    U_all = np.empty((len(N_spin), len(N_samples), len(T)))
    U_all_std = np.empty((len(N_spin), len(N_samples), len(T)))
    C_all = np.empty((len(N_spin), len(N_samples), len(T)))
    C_all_std = np.empty((len(N_spin), len(N_samples), len(T)))
    
    # Analytical result
    T_linspace = np.linspace(T[0], T[-1], 1000)
    beta_linspace = 1 / T_linspace
    
    U_theo = - np.tanh(beta_linspace)[:,np.newaxis] * (N_spin - 1) / N_spin
    C_theo = ((beta_linspace / np.cosh(beta_linspace))**2)[:, np.newaxis]
    C_theo *= (N_spin - 1) / N_spin
    
    colors = ["b", "g", "r"]
    marker = ["."]
    
    
    for j, N in enumerate(N_samples):
        fig_u, ax_u = plt.subplots(1,figsize=(5,4))
        fig_c, ax_c = plt.subplots(1,figsize=(5,4))

        ax_u.set_xlabel("$T$", fontsize=12)
        ax_u.set_ylabel("$U / N$", fontsize=12)
        ax_u.set_title("$N_{samples}$ = " + "{}".format(N))
        
        ax_c.set_xlabel("$T$", fontsize=12)
        ax_c.set_ylabel("$C / N$", fontsize=12)
        ax_c.set_title("$N_{samples}$ = " + "{}".format(N))
    
        for i, n in enumerate(N_spin):
            
            ax_u.plot(T_linspace, U_theo[:,i], color=colors[i], 
                      label="Theory $N_{spin} = $" + str(n))
        
            ax_c.plot(T_linspace, C_theo[:,i], color=colors[i], 
                      label="Theory $N_{spin} = $" + str(n))

                
            for k in range(T.shape[0]):    
              
                save_path = path + "N{0}_Nsample{1}_T{2:.1f}.npz".format(n, N, 
                                                                         T[k])
                
                data = np.load(save_path)
                E_tmp = data["Energy"][1:]
                data.close()
                
                # Here the physical qunatities are calculated as described in
                # in the report
                U_all[i,j,k] = np.mean(E_tmp)
                U_all_std[i,j,k] = np.std(E_tmp, ddof=1) / np.sqrt(N)
                C_all[i,j,k] = np.var(E_tmp, ddof=1) * beta[k]**2 / n
                
                U_all[i,j,k] /= n
                U_all_std[i,j,k] /= n
                C_all_std[i,j,k] = np.sqrt(2./(N)) * np.var(E_tmp, ddof=1)
                C_all_std[i,j,k] *= beta[k]**2 / n
    
            label = "$N_{spin} = $" + str(n)
            fmt = colors[i] + marker[0]
            ax_u.errorbar(T, U_all[i,j,:], yerr=U_all_std[i,j,:], 
                          fmt=fmt, label=label)
            ax_c.errorbar(T, C_all[i,j,:], yerr=C_all_std[i,j,:], 
                          fmt=fmt, label=label)
            
        ax_u.legend(fontsize=8)
        ax_u.grid(linestyle="--", alpha=0.5)
        ax_c.legend(fontsize=8)
        ax_c.grid(linestyle="--", alpha=0.5)
        fig_u.tight_layout()    
        fig_c.tight_layout() 
        fig_u.savefig(path + "U_N{}_1D.pdf".format(N))
        fig_c.savefig(path + "C_N{}_1D.pdf".format(N))
        fig_u.show() 
        fig_c.show()   
          
          
          
          
          
          
          
          
    