# -*- coding: utf-8 -*-
"""
Created on Thu May 21 12:28:20 2020
Python 3.7.7

@author: Christian Gorjaew, Julius Meyer-Ohlendorf
"""

import numpy as np
import numpy.random as rn
import h5py as h
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.constants import pi, Boltzmann, u

def lennard_jones(r_i, r_j, epsilon, sigma):
    """
    Calculates the Lennard-Jones potential for distance |r_i - r_j|.
    r_i, r_j: 1D numpy arrays of lenght dim where dim is the number of 
    dimensions.
    """
    r_ij = np.linalg.norm(r_i - r_j)
    
    sigma_r_ij_6 = (sigma / r_ij)**6
    
    return 4 * epsilon * (sigma_r_ij_6**2 - sigma_r_ij_6)

def acc_lennard_jones(r_i, r_j, m_i, epsilon, sigma):
    """
    Calculates the acceleration of particle at r_i due to particles at 
    positions r_j assuming Lennard-Jones-like force.
    r_i: 1D numpy arrays of lenght dim where dim is the number of 
    dimensions.
    r_j: Positions of all particles exerting force on r_i. Takes a ndarray
    of shape (M, dim) where M is the number of particles considered.
    """
    
    r_ij_vec = r_i - r_j

    r_ij = np.linalg.norm(r_ij_vec, axis=1)

    sigma_r_ij_6 = (sigma / r_ij)**6
    F_i_scalar = 24 * epsilon / r_ij**2 * (2*sigma_r_ij_6**2 - sigma_r_ij_6) 

    if np.any(np.isnan(r_ij)) or np.any(r_ij == 0):
        raise ValueError("NaN or divide by zero encountered. Aborting. Data will be saved.")

    F_i_scalar = F_i_scalar[:,np.newaxis]
    F_i = F_i_scalar * r_ij_vec
    F_i = np.sum(F_i, axis=0)
    return F_i / m_i

def wrap_r(file_, cell_dim, frame_ind=-1, make_plot=False,
           xlabel="$x$ [$\AA$]", ylabel="$y$ [$\AA$]"):
    """
    Wraps positions of atoms back into the intial cell. Assumes square cell. 
    Plots atom position if requested.
    file_: Path to data containing position data.
    cell_dim: Length of cell boundaries.
    frame_ind: Index of frame which is wrapped.
    """
    
    file_r = h.File(file_, "r")

    r = file_r["r"][frame_ind]
        
    file_r.close()
    
    N, dim = r.shape

    r_wrap = np.zeros((N, dim))

    # wrapping
    for j in range(N):
        cell_j = np.floor(r[j] / cell_dim)
        r_wrap[j] = r[j] - cell_j * cell_dim
        
    if make_plot:
        if dim != 2:
            raise RuntimeError("Plot not possible for dimensionality of data.")
        plt.figure(figsize=(4,3.5))
        plt.grid(linestyle="--", alpha=0.5)
        plt.scatter(r_wrap[:,0], r_wrap[:,1])
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.tight_layout()

    return r_wrap

def plot_quantitiy(path_ET, quantity_str, start=0, stop=None, 
                   save_name=None, figsize=(6,4)):
    """
    Creates plot of requested quantity with respect to time.
    path_ET: Path to file containing temperature and energies.
    quantity_str: String specifying qunatity that should be plotted (see 
    variable quantity_lut).
    """
    if quantity_str != "T":    
        k_B = Boltzmann / u * 1e10**2 / 1e12**2 
    else:
        k_B = 1
        
    quantity_lut = {"T": 1, "E_tot": 2, "E_pot": 3, "E_kin": 4}
    ylabel_lut = {"T": "$T$ [K]", "E_tot": 
                  "$E_{tot} / k_B$ [K]", 
                  "E_pot": "$E_{pot} / k_B$ [K]", 
                  "E_kin": "$E_{kin} / k_B$ [K]"}
    
    with h.File(path_ET, "r") as file_ET:
        time = file_ET["E_T"][start:stop, 0]
        quantity = file_ET["E_T"][start:stop, quantity_lut[quantity_str]]
        file_ET.close()
    
    plt.figure(figsize=figsize)
    plt.plot(time, quantity / k_B)
    plt.xlabel("time [ps]", fontsize=12)
    plt.ylabel(ylabel_lut[quantity_str], fontsize=12)
    plt.grid()
    plt.title(ylabel_lut[quantity_str] + " for $t$ = {0:.2f}..{1:.2f}$~$ps".format(time[0], time[-1]), 
              fontsize=12)
    plt.tight_layout()
    if save_name is not None:
        plt.savefig(save_name)
        plt.close()

def calc_fMB(path_v, save_name=None, start=-10,
             stop=None, T=150, m=39.9, k_B=0.83):
    '''
    Calculate speed distribution for frames "start" to "stop". Default: plots 
    last 10 frames.
    '''
    # calculating theoretical distribution
    x_max = 8
    vMB = np.linspace(0, x_max, 100)
    fMB_theo = 2 * pi * m / (2 * pi * k_B * T) * vMB * np.exp(-m * vMB**2 /
              (2* k_B * T))

    # calculating distribution from data
    with h.File(path_v, "r") as file_v:
        v = file_v["v"][:]
        time = file_v["t"][:]
        file_v.close()

    v_pick = v[start:stop]
    v_hist = np.linalg.norm(v_pick, axis=2)
    v_hist = v_hist.flatten()
    title = 'Speed distribution {0} frames ({1:.2f}ps to {2:.2f}ps)'.format(
             v_pick.shape[0], time[start], time[-1 if stop is None else stop])

    plt.figure()
    plt.title(title, fontsize=10)
    plt.hist(v_hist, bins=40, density=True, label='speed distribution')
    plt.plot(np.linspace(0, x_max, 100), fMB_theo, 'r',
             label='MB distribution 2D for $T~=~${0}$~$K'.format(T))
    plt.xlabel('|v|[$\AA$/ps]', fontsize=12)
    plt.ylabel('relative frequency', fontsize=12)
    plt.legend(loc='upper right')
    if save_name is not None:
        plt.savefig('./plots/{0}_Nframes{1}.pdf'.format(
                    save_name, v_pick.shape[0]), bbox_inches='tight')
        plt.close()

    return(v_hist, fMB_theo)

def pair_correlation(file_, density, dr, start=0, stop=None, make_plot=False,
                     xlabel="$r$ [$\AA$]", ylable="$g(r)$", label=None):
    """
    Calculates radial pair correlation function for given position values.
    Wraps the positions back to the initial cell before the calculation.
    """
    
    
    data = h.File(file_, "r")
    r_vec = data["r"][start:stop]
    
    n_steps = r_vec.shape[0]
    N = r_vec.shape[1]
    dim = r_vec.shape[2]
    
    cell_dim = np.sqrt(float(N) / density)
    
    r_wrap = np.zeros_like(r_vec)
    
    for i in range(n_steps):
        for j in range(N):
            cell_j = np.floor(r_vec[i][j] / cell_dim) * 0
            r_wrap[i][j] = r_vec[i][j] - cell_j * cell_dim
        
    print("Using ", n_steps, " time steps")
    r_max = np.sqrt(N / density) / 2
    r_bins = np.arange(0., r_max + dr, dr)  # Intervals for [k*dr, (k+1*dr]
    r_k = r_bins[0:-1] + dr / 2
    
    r_ij = np.zeros((n_steps, N*(N-1), 2))
    
    # Calculate distances between al pairs for all frames
    for step in range(n_steps):
        ind = 0
        for i in range(N):
            for j in range(N):
                if i != j:
                    r_ij[step][ind] = r_wrap[step][i] - r_wrap[step][j]
                    ind += 1
    
    r_ij = np.linalg.norm(r_ij, axis=2)
                    
    
    counts = np.histogram(r_ij.flatten(), bins=r_bins)[0]
    h_mean = counts.astype(float) / n_steps
    
    if dim == 2:
        A = 2 * np.pi * r_k     # Denominator for 2D
    elif dim == 3:
        A = 4 * np.pi * r_k**2  # Denominator for 3D
    else:
        raise RuntimeError("Not applicable due to dimensionality.")
    
    g = 2 * h_mean / density / (N-1) / A / dr
    
    normalization = np.sum(A * g * dr)  # Check normalization
    
    if make_plot:
        plt.figure(figsize=(6,3.5))
        plt.plot(r_k, g, label=label)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylable, fontsize=12)
        plt.grid()
        if label is not None:
            plt.legend(fontsize=10)
        plt.tight_layout()
        plt.show()
    
    return g, r_k, normalization
    
    
class MDEngine(object):
    
    def __init__(self, dim, N, cell_dim, potential, acceleration, temperature, 
                 masses, r_init, tau, k_B=Boltzmann ,path="./", **kwargs):
        
        self.dim = dim              # Dimension of system
        self.N = N                  # Total number of particles
        self.cell = cell_dim        # Dimensions of the initial cell (float for square cell, (dim,) shaped array otherwise)
        self.pot = potential        # Potential function describing  a pair potential. Takes r_i, r_j, **kwargs
        self.acc = acceleration     # Function calculating total acceleration of a particle. Takes r_i, r_j (position of all other particles), m_i, **kwargs
        self.T = temperature        # Target temperatur
        self.tau = tau              # Time step size
        self.m = masses             # Masses of all particles. (N,1) array
        self.r = r_init             # (Initial) particle positions. (N,dim) array
        self.v = None               # Velocities. Initialized in "run_nvt". (N, dim) array
        self.t = 0.                 # Current time
        self.step = 0               # Current step
        self.E_pot = None           # Working variable for potential energy
        self.E_kin= None            # Working variable for kinetic energy
        self.T_current = None       # Working variable for instantaneous temperature
        self.path = path            # Path to folder where simulation data is stored
        self.existing_data = False
        
        self.k_B = k_B              # Redifintion of Boltzmann constant in case of non SI units are used

        self._kwargs = kwargs       # Keyword arguments for potential and acceleration functions
        
        self._cell_steps = self._gen_cell_steps()
        
        self._verify_data_shape()
        
    def _gen_cell_steps(self):
        """
        Generates shift arrays that are used to find the nearest neighbor
        (see self.find_nearest_neighbor()) of an particle i and an (image) 
        particle j accourding to the minimum image convention.
        Outputs a (27, 3) shaped array containg integer steps from the 
        cell under consideration into the neighboring cells.
        Cell under consideration (contains particle i) is assigned to 
        entry [0, 0, 0].
        
        For a 1D system, only the first 3 entries and the 1. column will be 
        used in self.find_nearest_neighbor().
        For a 2D system, only the first 9 entries and the 1. and 2. column will
        be used.
        For a 3D system, all entries will be used.
        """
        cell_steps = []
        for kz in range(-1,2,1):
            for ky in range(-1,2,1):
                for kx in range(-1,2,1):
                    cell_steps += [[kx, ky, kz]]
        return np.array(cell_steps)
        
    def _verify_data_shape(self):
        try:
            self.m.shape[1]
        except IndexError:
            self.m = self.m[:,np.newaxis]
            
        try:
            self.r.shape[1]
        except IndexError:
            assert(self.r.shape[0] == self.N * self.dim)
            self.r = self.r.reshape((self.N, self.dim))
        
    def _init_velocity(self):
        """
        Initialize velocities folloring Maxwell-Boltzmann distribution and 
        shift to zero CMS movement.

        Returns
        -------
        None.

        """
        v_tmp = np.zeros((self.N,self.dim))
        
        for i, m in enumerate(self.m):
            width = np.sqrt(self.k_B * self.T / m)
            sample_norm = rn.normal(0.0, width, self.dim)
            v_tmp[i] = sample_norm
        
        P = np.sum(self.m * v_tmp, axis=0)  # Total momentum
        
        self.v = v_tmp - P / self.m / self.N
        
    
    def velocity_verlet_step(self):
        """"
        Executes one step of the velocity verlet method.
        
        Acceleration function takes the positions as flattened array.
        """
        r_old = self.r
        v_old = self.v
        t = self.t
        acc_old = self.get_acc(r_old, t)
        r_new = r_old + self.tau * v_old + 0.5 * self.tau**2 * acc_old
        
        acc_new = self.get_acc(r_new, t)
        v_new = v_old + 0.5 * self.tau * (acc_old + acc_new)
        
        self.t += self.tau
        
        return r_new, v_new
    
    def calc_E_kin(self):
        
        E_kin = 0.5 * self.m * self.v**2
        
        return E_kin.sum()
    
    def calc_E_pot(self):
        """ 
        Calculates potential energy of the system with wrapped coordinates.
        """
        r_wrap = self.wrap_r()
        E_pot = 0
        for i in range(self.N):
            for j in range(i+1, self.N):
                E_pot += self.pot(r_wrap[i], r_wrap[j], **self._kwargs)
                
        return E_pot
    
    def calc_T_instant(self, E_kin):
        
        return 2 * E_kin / self.dim / self.k_B / (self.N - 1)
        

    def find_nearest_neighbor(self, part_ind):
        """
        Finds all nearest neighbors of a particle described by index 'part_ind'
        accourding to the minimum image convention. I.e., all particles, resp.
        their images, with index j != part_ind 
        """
        
        r_nearest = np.zeros((self.N-1, self.dim))
        nearest_ind = 0
        
        for j in range(self.N):
            if j != part_ind:
                
                cell_i = np.floor(self.r[part_ind] / self.cell)  # cell index containing particle under consideration
                # For example: 2D system: particle leaves initial cell through left border --> cell_i = [-1, 0]
                
                cell_j = np.floor(self.r[j] / self.cell)
                
                cell_diff = cell_j - cell_i
            
                r_j_cell_i = self.r[j] - cell_diff * self.cell  # position of (image of) particle j in cell_i
                n_adj_cells = 3**self.dim  # number of adjacent cells including cell_i
                
                r_j_adj = np.zeros((n_adj_cells, self.dim))
                
                for k in range(n_adj_cells):
                    # positions of images of particle j in cells adjacent to cell_i
                    r_j_adj[k] = r_j_cell_i + self._cell_steps[k,0:self.dim] * self.cell
                
                diff_adj_vec = self.r[part_ind] - r_j_adj  # distance between particle 'part_ind' and all images of j in adjacent cells and cell_i
                diff_adj = np.linalg.norm(diff_adj_vec, axis=1)
                ind_min_dist = np.argmin(diff_adj)
                
                r_nearest[nearest_ind] = r_j_adj[ind_min_dist]
                nearest_ind += 1

        return r_nearest
    
    def get_acc(self, r, t):
        """
        Calculates acceleration for all particles using the given acceleration
        function using minimum image convention.
        """
        acc = np.zeros_like(r)
        
        for i in range(self.N):
            r_j_nearest = self.find_nearest_neighbor(i)
            acc[i] = self.acc(r[i], r_j_nearest, self.m[i], **self._kwargs)
        
        return acc
    
    def wrap_r(self):
        """
        Returns a wrapped coordinates of the current position.
        """
        r_wrap = np.zeros((self.N, self.dim))

        # wrapping
        for j in range(self.N):
            cell_j = np.floor(self.r[j] / self.cell)
            r_wrap[j] = self.r[j] - cell_j * self.cell

        return r_wrap
    
    def load_frame(self, path, nvt_data=True, frame=-1):
        """
        Loads the last positions, velocities, step number and time contained
        in position and velocity data files produced by class methods
        self.run_nvt() or self.run_nve().
        path: Path to directory containing data files.
              Position file must have the name "r_nvt.hdf5" or "r_nve.hdf"
              Velocity file must have the name "v_nvt.hdf5" or "v_nve.hdf"
        nvt_data: Specify True if data is NVT equilibration data.
                  False if data was produced in NVE production run.
        """
        if nvt_data:
            appendix = "nvt"
        else:
            appendix = "nve"
        self.path = path
        path_r = path + "r_" + appendix + ".hdf5"
        path_v = path + "v_" + appendix + ".hdf5"
        
        file_r, file_v = h.File(path_r, "r"), h.File(path_v, "r")
        n_r, n_v = file_r["step"][frame], file_v["step"][frame]
        assert (n_r == n_v)
        print("Using frame index " + str(frame) + "corresponding to step " + str(n_r))
        self.step = n_r
        self.t = file_r["t"][frame]
        self.r = file_r["r"][frame]
        self.v = file_v["v"][frame]
        file_r.close()
        file_v.close()
        
        self.existing_data = True
    
    def write_energy_temp(self, file_):
        
        E_tot = self.E_pot + self.E_kin
        to_save = np.array([self.t, self.T_current, E_tot, self.E_pot, self.E_kin])
        file_["E_T"][self.step] = to_save
        file_["step"][self.step] = self.step
    
    def write_r(self, file_, storage_frame):
        
        ind = int(self.step / storage_frame)
        file_["step"][ind] = self.step
        file_["t"][ind] = self.t
        file_["r"][ind] = self.r
    
    def write_v(self, file_, storage_frame):
        
        ind = int(self.step / storage_frame)
        file_["step"][ind] = self.step
        file_["t"][ind] = self.t
        file_["v"][ind] = self.v
    
    def run_nvt(self, n_steps):
        """
        Runs 'n_steps' NVT equilibration steps using velocity rescaling.
        Rescaling is applied every 10 steps. 
        Storage of positions and velocities every 10th step, for temperature,
        and energies every step.
        hdf5-files are used for storage.
        Outputs data files "r_nvt.hdf5" (positions, time, steps), "v_nvt.hdf5"
        (velocities, time, steps) and "Energy_Temp_nvt.hdf5" (steps, time,
        instant. temperature, total energy, potential energy, kinetic energy).
        Appends to those files if data was loaded manually.
        """
        storage_frame = 10
        assert ((n_steps % storage_frame) == 0)
        
        
        with h.File(self.path + "r_nvt.hdf5", "a") as file_r, h.File(self.path + "v_nvt.hdf5", "a") as file_v, h.File(self.path + "Energy_Temp_nvt.hdf5", "a") as file_ET: 

        
            if self.v is None:
                self._init_velocity()  # Initializing velocities
                
                # Creating file structure of output files if not existing yet
                entries_rv = int(n_steps/storage_frame) + 1  # pos. & vel. have less entries due to different storage frame
                file_r.create_dataset("step", shape=(entries_rv,), maxshape=(None,), chunks=True ,dtype=int)
                file_r.create_dataset("t", shape=(entries_rv,), chunks=True , maxshape=(None,))
                file_r.create_dataset("r", shape=(entries_rv, self.N, self.dim), chunks=True , maxshape=(None, self.N, self.dim))
                
                file_v.create_dataset("step", shape=(entries_rv,), maxshape=(None,), chunks=True , dtype=int)
                file_v.create_dataset("t", shape=(entries_rv,), chunks=True , maxshape=(None,))
                file_v.create_dataset("v", shape=(entries_rv, self.N, self.dim), chunks=True , maxshape=(None, self.N, self.dim))
                
                file_ET.create_dataset("step", shape=(n_steps + 1,), chunks=True, maxshape=(None,), dtype=int)
                file_ET.create_dataset("E_T", shape=(n_steps + 1, 5), chunks=True , maxshape=(None, 5))
                
                
                self.E_kin = self.calc_E_kin()
                self.E_pot = self.calc_E_pot()
                self.T_current = self.calc_T_instant(self.E_kin)
                
                
                self.write_energy_temp(file_ET)
                self.write_r(file_r, storage_frame)
                self.write_v(file_v, storage_frame)
                
                
            
            if self.existing_data:
                # resizes output data files to append new data
                new_entries_rv = int(n_steps/storage_frame)

                for key in file_r.keys(): 
                    file_r[key].resize(file_r[key].shape[0] + new_entries_rv, axis=0)
                for key in file_v.keys():
                    file_v[key].resize(file_v[key].shape[0] + new_entries_rv, axis=0)
                for key in file_ET.keys():
                    file_ET[key].resize(file_ET[key].shape[0] + n_steps, axis=0)

            self.existing_data = True
            
            for i in tqdm(range(1,n_steps + 1)):
                print("Step " + str(i))
                self.r, self.v = self.velocity_verlet_step()
            
                if (i % 10) == 0:
                    # Velocity rescaling
                    Lambda = np.sqrt(self.dim*self.k_B*(self.N-1)*self.T / 2 / self.calc_E_kin())
                    self.v *= Lambda
                
                self.step += 1

                self.E_kin = self.calc_E_kin()
                self.E_pot = self.calc_E_pot()
                self.T_current = self.calc_T_instant(self.E_kin)
                
                self.write_energy_temp(file_ET)
                
                if (i % storage_frame) == 0:
                    self.write_r(file_r, storage_frame)
                    self.write_v(file_v, storage_frame)
            
                    
                
    def run_nve(self, n_steps, nve_exists=False):
        """
        NVE production run. As self.run_nvt() but without velocity rescaling.
        Velocities are shifted in the beginning such that total momentum is 
        zero.
        Output into files "r_nve.hdf5", "v_nve.hdf5" and "Energy_Temp_nve.hdf5".
        Number of equilibration steps is saved in data files if data from 
        equilibration run is used (leave nve_exists as False).
        """
        if self.v is None or not self.existing_data:
            raise RuntimeError("No data loaded. Load equilibrated data with MDEngine.load_frame.")
        storage_frame = 10
        
        
        # Shifting velocities such that P_tot = 0
        P_tot = np.sum(self.m * self.v, axis=0)  # Total momentum
        
        self.v = self.v - P_tot / self.m / self.N
        print(np.sum(self.v,axis=0))
        
        with h.File(self.path + "r_nve.hdf5", "a") as file_r, h.File(self.path + "v_nve.hdf5", "a") as file_v, h.File(self.path + "Energy_Temp_nve.hdf5", "a") as file_ET: 

        
            if not nve_exists:
                
                entries_rv = int(n_steps/storage_frame) + 1
                file_r.create_dataset("n_eqil", data=self.step)
                file_r.create_dataset("step", shape=(entries_rv,), maxshape=(None,), chunks=True ,dtype=int)
                file_r.create_dataset("t", shape=(entries_rv,), chunks=True , maxshape=(None,))
                file_r.create_dataset("r", shape=(entries_rv, self.N, self.dim), chunks=True , maxshape=(None, self.N, self.dim))

                file_v.create_dataset("n_eqil", data=self.step)
                file_v.create_dataset("step", shape=(entries_rv,), maxshape=(None,), chunks=True , dtype=int)
                file_v.create_dataset("t", shape=(entries_rv,), chunks=True , maxshape=(None,))
                file_v.create_dataset("v", shape=(entries_rv, self.N, self.dim), chunks=True , maxshape=(None, self.N, self.dim))
                
                file_ET.create_dataset("n_eqil", data=self.step)
                file_ET.create_dataset("step", shape=(n_steps + 1,), chunks=True, maxshape=(None,), dtype=int)
                file_ET.create_dataset("E_T", shape=(n_steps + 1, 5), chunks=True , maxshape=(None, 5))
                self.step = 0
                
                self.E_kin = self.calc_E_kin()
                self.E_pot = self.calc_E_pot()
                self.T_current = self.calc_T_instant(self.E_kin)
                
                self.write_energy_temp(file_ET)
                self.write_r(file_r, storage_frame)
                self.write_v(file_v, storage_frame)
            else:
                new_entries_rv = int(n_steps/storage_frame)
    
                for key in file_r.keys(): 
                    if not key == "n_eqil":
                        file_r[key].resize(file_r[key].shape[0] + new_entries_rv, axis=0)
                for key in file_v.keys():
                    if not key == "n_eqil":
                        file_v[key].resize(file_v[key].shape[0] + new_entries_rv, axis=0)
                for key in file_ET.keys():
                    if not key == "n_eqil":
                        file_ET[key].resize(file_ET[key].shape[0] + n_steps, axis=0)   
            
            for i in tqdm(range(1,n_steps + 1)):
                self.r, self.v = self.velocity_verlet_step()
                print("Step " + str(i))
                self.step += 1

                self.E_kin = self.calc_E_kin()
                self.E_pot = self.calc_E_pot()
                self.T_current = self.calc_T_instant(self.E_kin)
                
                self.write_energy_temp(file_ET)
                
                if (i % storage_frame) == 0:
                    self.write_r(file_r, storage_frame)
                    self.write_v(file_v, storage_frame)
            
                
                
