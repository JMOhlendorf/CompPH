import numpy as np
import cp_ex05_engine as eng
import matplotlib.pyplot as plt
import os
import h5py as h

from scipy.constants import u, pi, Boltzmann

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
cell_dim1 = np.sqrt(N / 0.07)  # 37.8     # corresponding to rho_1
cell_dim2 = np.sqrt(N / 0.1)   # 31.6     # corresponding to rho_2

# epsilon, sigma and k_B in new units
epsilon = 1.65e-21 / u * 1e10**2 / 1e12**2  # ~ 99.4
sigma = 3.4e-10 * 1e10
k_B = Boltzmann / u * 1e10**2 / 1e12**2 # ~ 0.83

# initial atomic configuration
r_init1 = r_initializer(cell_dim1, dim)

path = './simulation_jmo/rho1/'


# plotting initial atomic configuration for rho1
###########################################

plt.figure(figsize=(4,3.5))
plt.grid(linestyle="--", alpha=0.5)
plt.scatter(r_init1[:,0], r_init1[:,1])
plt.xlabel('$x$ [$\AA$]', fontsize=12)
plt.ylabel('$y$ [$\AA$]', fontsize=12)
plt.xlim(0, cell_dim1)
plt.ylim(0, cell_dim1)
plt.savefig(path + 'plots/config_initial_rho1_NVT.pdf', bbox_inches='tight')
plt.show()
plt.close()


# NVT
###########################################
###########################################

# equilibration NVT run for rho_1 configuration
###########################################

engine1 = eng.MDEngine(dim=dim, N=N, cell_dim=cell_dim1, potential=potential,
                       acceleration=acceleration, temperature=temperature,
                       masses=masses, r_init=r_init1, tau=tau, k_B=k_B,
                       path=path, epsilon=epsilon, sigma=sigma)

engine1.run_nvt(n_steps=1500)



# plotting Energy and Temp
###########################################
nsteps_nvt = 1000

eng.plot_quantitiy(path + 'Energy_Temp_nvt.hdf5', quantity_str="T", stop=nsteps_nvt,
                   save_name=path + 'plots/T_nsteps{0}_rho1_NVT.pdf'.format(nsteps_nvt))

eng.plot_quantitiy(path + 'Energy_Temp_nvt.hdf5', quantity_str="E_tot", stop=nsteps_nvt,
                   save_name=path + 'plots/Etot_nsteps{0}_rho1_NVT.pdf'.format(nsteps_nvt))

r_wrap = eng.wrap_r(path + 'r_nvt.hdf5', cell_dim=cell_dim1, frame_ind=-1, make_plot=False)

plt.figure(figsize=(4,3.5))
plt.grid(linestyle="--", alpha=0.5)
plt.scatter(r_wrap[:,0], r_wrap[:,1])
plt.xlabel('$x$ [$\AA$]', fontsize=12)
plt.ylabel('$y$ [$\AA$]', fontsize=12)
plt.xlim(0, cell_dim1)
plt.ylim(0, cell_dim1)
plt.savefig(path + 'plots/config_nsteps{0}_rho1_NVT.pdf'.format(nsteps_nvt),
            bbox_inches='tight')
plt.show()
plt.close()


eng.calc_fMB(path + 'v_nvt.hdf5', save_name='rho1_NVT_nsteps{0}'.format(nsteps_nvt), start=-81, stop=-51)


# NVE
###########################################
###########################################

# production NVE run for rho_1 configuration
###########################################
engine1 = eng.MDEngine(dim=dim, N=N, cell_dim=cell_dim1, potential=potential,
                       acceleration=acceleration, temperature=temperature,
                       masses=masses, r_init=r_init1, tau=tau, k_B=k_B,
                       path=path, epsilon=epsilon, sigma=sigma)

path_reload = './simulation_jmo/rho1/'

engine1.load_frame(path_reload, nvt_data=True, frame=100)
engine1.run_nve(n_steps=500)



# plotting Energy and Temp
###########################################
nsteps_nve = 500

eng.plot_quantitiy(path + 'Energy_Temp_nve.hdf5', quantity_str="T", stop=nsteps_nve,
                   save_name=path + 'plots/T_nsteps{0}_rho1_NVE.pdf'.format(nsteps_nve))

eng.plot_quantitiy(path + 'Energy_Temp_nve.hdf5', quantity_str="E_tot", stop=nsteps_nve,
                   save_name=path + 'plots/Etot_nsteps{0}_rho1_NVE.pdf'.format(nsteps_nve))

r_wrap = eng.wrap_r(path + 'r_nve.hdf5', cell_dim=cell_dim1, frame_ind=-1, make_plot=False)

plt.figure(3)
plt.grid(linestyle="--", alpha=0.5)
plt.scatter(r_wrap[:,0], r_wrap[:,1])
plt.xlabel('$x$ [$\AA$]', fontsize=12)
plt.ylabel('$y$ [$\AA$]', fontsize=12)
plt.savefig(path + 'plots/config_nsteps{0}_rho1_NVE.pdf'.format(nsteps_nve),
            bbox_inches='tight')
plt.show()
plt.close()

eng.calc_fMB(path + 'v_nve.hdf5', save_name='rho1_NVE_nsteps{0}'.format(nsteps_nve), start=-30)
