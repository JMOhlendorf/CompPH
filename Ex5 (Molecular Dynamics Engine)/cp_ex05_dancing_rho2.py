import numpy as np
import cp_ex05_engine as eng
import matplotlib.pyplot as plt
import os

from scipy.constants import u, Boltzmann
# function initializing atomic positions
def r_initializer(cell_dim, dim):
    r_init = []
    spacing = cell_dim / 10
    x = np.arange(0, cell_dim, spacing)
    y = np.arange(0, cell_dim, spacing)
    for ele_y in y:
        for ele_x in x:
            if dim == 2:
                r_init.append([ele_x, ele_y])
            else:
                r_init.append([ele_x, ele_y, 0])
    return(np.array(r_init))

# some simulation parameters
dim = 2
N = 100
potential = eng.lennard_jones
acceleration = eng.acc_lennard_jones
temperature = 150
tau = 0.01

# masses
mass_Ar = 39.9
masses = np.empty((N))
masses.fill(39.9)

# length cell_dim of simulation box given by cell_dim=(100/rho)^(1/2)
rho_2 = 0.1
cell_dim2 = np.sqrt(N / rho_2)   # 31.6     # corresponding to rho_2

# epsilon, sigma and k_B in new units
epsilon = 1.65e-21 / u * 1e10**2 / 1e12**2  # ~ 99.4
sigma = 3.4e-10 * 1e10
k_B = Boltzmann / u * 1e10**2 / 1e12**2 # ~ 0.83

# initial atomic configuration
r_init2 = r_initializer(cell_dim2, dim)


#%%
# equilibration run for rho_2 configuration
###########################################
dir_name = "./rho_2/"

# plotting initial atomic configuration
###########################################


engine1 = eng.MDEngine(dim=dim, N=N, cell_dim=cell_dim2, potential=potential,
                       acceleration=acceleration, temperature=temperature,
                       masses=masses, r_init=r_init2, tau=tau, k_B=k_B,
                       path=dir_name, epsilon=epsilon, sigma=sigma)
#%%%
engine1.run_nvt(n_steps=1500)
eng.plot_quantitiy(dir_name + "Energy_Temp_nvt.hdf5", "T", stop=1000, save_name="./rho_2/equil_time_vs_T.pdf")
eng.plot_quantitiy(dir_name + "Energy_Temp_nvt.hdf5", "E_tot", stop=1000, save_name="./rho_2/equil_time_vs_E_tot.pdf")
eng.plot_quantitiy(dir_name + "Energy_Temp_nvt.hdf5", "E_pot", stop=1000, save_name="./rho_2/equil_time_vs_E_pot.pdf")
eng.plot_quantitiy(dir_name + "Energy_Temp_nvt.hdf5", "E_kin", stop=1000, save_name="./rho_2/equil_time_vs_E_kin.pdf")
#%%
eng.wrap_r(dir_name + "r_nvt.hdf5", cell_dim=cell_dim2, make_plot=True, frame_ind=0)
eng.wrap_r(dir_name + "r_nvt.hdf5", cell_dim=cell_dim2, make_plot=True, frame_ind=100)
eng.calc_fMB(dir_name + "v_nvt.hdf5", save_name="rho_2_equil", start=71, stop=101, k_B=k_B)

#%%
engine1.load_frame(dir_name, nvt_data=True, frame=100)
try:
    engine1.run_nve(n_steps=500)
except ValueError:
    print("Stop after {0} NVE steps.".format(engine1.step))
    plt.figure(figsize=(4,3.5))
    plt.grid(linestyle="--", alpha=0.5)
    plt.scatter(engine1.r[:,0], engine1.r[:,1])
    plt.xlabel("x [$\AA$]", fontsize=12)
    plt.ylabel("y [$\AA$]", fontsize=12)
    plt.tight_layout()
    last_step = engine1.step
#%%
eng.plot_quantitiy(dir_name + "Energy_Temp_nve.hdf5", "T", stop=last_step, save_name="./rho_2/nve_time_vs_T.pdf")
eng.plot_quantitiy(dir_name + "Energy_Temp_nve.hdf5", "E_tot", stop=last_step, save_name="./rho_2/nve_time_vs_E_tot.pdf")
eng.plot_quantitiy(dir_name + "Energy_Temp_nve.hdf5", "E_pot", stop=last_step, save_name="./rho_2/nve_time_vs_E_pot.pdf")
eng.plot_quantitiy(dir_name + "Energy_Temp_nve.hdf5", "E_kin", stop=last_step, save_name="./rho_2/nve_time_vs_E_kin.pdf")
#%%
eng.wrap_r(dir_name + "r_nve.hdf5", cell_dim=cell_dim2, make_plot=True, frame_ind=int(float(last_step)/10))
eng.calc_fMB(dir_name + "v_nve.hdf5", save_name="rho_2_nve", start=0, stop=int(float(last_step)/10), k_B=k_B)

#%% Calculation of pair correlation functions for rho_1 and rho_2
#################################################################
rho_1 = 0.07
rho_2 = 0.1
g_1_nvt, r_k_1_nvt, norm_1 = eng.pair_correlation("./rho_1/r_nvt.hdf5",
                                                  rho_1, dr=0.1, start=20, stop=101)
g_2_nvt, r_k_2_nvt, norm_2 = eng.pair_correlation(dir_name + "r_nvt.hdf5",
                                                  rho_2, dr=0.1, start=20, stop=101)
plt.figure(figsize=(6,3.5))
label_1 = "$\\rho_1~=~${0}$~\AA^2$\nNormalization$~=~$ {1:.2f}$~\AA$".format(rho_1, norm_1)
label_2 = "$\\rho_2~=~${0}$~\AA^2$\nNormalization$~=~$ {1:.2f}$~\AA$".format(rho_2, norm_2)
plt.plot(r_k_1_nvt, g_1_nvt, label=label_1)
plt.plot(r_k_2_nvt, g_2_nvt, label=label_2)
plt.xlabel("$r$ [$\AA$]", fontsize=12)
plt.ylabel("$g(r)$", fontsize=12)
plt.grid()
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()