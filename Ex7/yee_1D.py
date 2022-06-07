"""
Python 3.7.7

@author: Christian Gorjaew, Julius Meyer-Ohlendorf
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import h5py as h5

from tqdm import tqdm
from numpy import sin, exp, pi

def yee_step(E, H, l_s, J, A, B, C, D, delta):
    """
    Updates the Yee grid by one time step
    """
    E[1:-1] *= C[1:-1]
    E[1:-1] +=  D[1:-1] * (H[1:] - H[0:-1]) / delta
    E[l_s] -= D[l_s] * J
    H *= A
    H +=  B * (E[1:] - E[0:-1]) / delta

"""
###### Defining physical and simulation parameters on grid #####
###### Change parameters here ####
"""

L = 5000
m = 10000

lamb = 1.
delta = lamb / 50
step_fac = 0.9  # Change here for different time step

n_refr = 1.46

tau = step_fac * delta
tarr = np.arange(0, m * tau, tau)  # array with time values

# Setting sigma(x)
left_bound = int(6 * lamb / delta)
sigma = np.zeros(L + 1)
sigma[0: left_bound] = 1.
sigma[L - left_bound:] = 1.

# setting sigma*(x)
sigma_star = np.zeros(L)
sigma_star[0: left_bound-1] = 1.
sigma_star[L - left_bound:] = 1.

# setting permittivity epsilon(x)
left_bound_eps = int(np.ceil(L / 2))  # Change to -1 for thick glas plate
right_bound_eps = int(np.ceil(L / 2 + 2 * lamb / delta))
eps = np.ones(L + 1)
eps[left_bound_eps:right_bound_eps] = n_refr**2

# setting magnetic permeability mu(x)
mu = np.ones(L)

# setting parameters A, B, C, D employed in Yee steps
A = (1 - sigma_star * tau / 2 / mu) / (1 + sigma_star * tau / 2 / mu)
B = tau / mu / (1 + sigma_star * tau / 2 / mu)
C = (1 - sigma * tau / 2 / eps) / (1 + sigma * tau / 2 / eps)
D = tau / eps / (1 + sigma * tau / 2 / eps)

# setting source for whole duration of simulation in time steps of tau
x_s = 20 * lamb
i_s = int(x_s / delta)
J_s = sin(2 * pi / lamb * tarr) * exp(-((tarr - 30.) / 10.)**2)

# initializing E and H fields with zeros
E = np.zeros(L + 1)
H = np.zeros(L)

# setting up storage frame
storage_frame = 50
storage_steps = int(m / storage_frame)

# storage path and file name, change 'extra_string' for different configuration
extra_string = "_0.9_thin"  
path = "./yee_data" + extra_string + ".hdf5"

#%%  <-- signature lets you excecute code cells with some python interpreters

# excecuting the actual simulation
with h5.File(path, "a") as file:
    # setting up the storage file of type 'hdf5'
    file.create_dataset("step", shape=(storage_steps,), maxshape=(None,), 
                        chunks=True, dtype=int)
    file.create_dataset("t", shape=(storage_steps,), maxshape=(None,), 
                        chunks=True, dtype=float)
    file.create_dataset("E", shape=(storage_steps, L+1), maxshape=(None, L+1),
                        chunks=True, dtype=float)
    file.create_dataset("H", shape=(storage_steps, L), maxshape=(None, L),
                        chunks=True, dtype=float)
    k_storage = 0
    
    # actual simulation loop, 'try-except' to catch exception due to divergence
    # when Courant condition is not met
    try:
        for step in tqdm(range(m)):
            
            yee_step(E, H, i_s, J_s[step], A, B, C, D, delta)
            
            if (step % storage_frame) == 0:
                file["step"][k_storage] = step
                file["t"][k_storage] = tarr[step]
                file["E"][k_storage] = E
                file["H"][k_storage] = H
    
                k_storage += 1
    except:
        pass

#%%
xx = np.arange(0., (L+1)*delta, delta)  # simulation domain

def animate_E(path, extra_string=""):
    """
    Generates a gif displaying the solution for the electric field that was
    simulated with the Yee algorithm above
    """
    fig, ax = plt.subplots(figsize=(5,3.5))
    fig.set_tight_layout(True)
    
    
    lineE, = ax.plot(xx, np.zeros_like(xx), linewidth=0.75)    
    
    def update(i):
        step = int(t[i] / tau)
        title = "$t / \\tau = ${0} ($\\tau = ${1}$\Delta$)".format(step, step_fac)
        lineE.set_ydata(E_d[i])
        ax.set_title(title, fontsize=12)
        
        return (lineE, ax)
        
    data = h5.File(path, "r")
    
    t = data["t"][:]
    E_d = data["E"][:]
    data.close()
    ax.set_xlim(0,xx[-1])
    ax.set_ylim(np.min(E_d)*1.05, np.max(E_d)*1.05)
    ax.set_xlabel("$x$", fontsize=12)
    ax.set_ylabel("$E$", fontsize=12)
    ax.axvspan(xx[left_bound_eps], xx[right_bound_eps], color="g", alpha=0.3,
               label="$n = ${0}, $\\sigma = \\sigma^* = 0$".format(n_refr))
    ax.axvspan(0., xx[left_bound], color="grey", alpha=0.3, 
               label="$n = 1$, $\\sigma = \\sigma^* = 1$".format(n_refr))
    ax.axvspan(xx[L - left_bound], xx[-1], color="grey", alpha=0.3)
    ax.grid(linestyle="--", alpha=0.5)
    ax.legend(loc=1, fontsize=10)
    
    anim = FuncAnimation(fig, update, frames=np.arange(t.shape[0]))
    anim.save("E" + extra_string + ".gif", dpi=80)
    
animate_E(path, extra_string)

#%% 
# calculation of the reflectivity
if extra_string == "_0.9_thick":
    # setting time steps at which incident and reflected wave are clearly 
    # visible
    step_inc = 2400
    step_refl = 4500
    
    # loading data
    data = h5.File(path, "r")
    steps = data["step"][:]
    E_d = data["E"][:]
    ind_inc = np.where(steps == step_inc)[0]
    ind_refl = np.where(steps == step_refl)[0]
    
    # determining maximum of absolute values of the waves
    E_inc_max = np.max(np.abs(E_d[ind_inc][0,0:left_bound_eps]))
    E_refl_max = np.max(np.abs(E_d[ind_refl][0,0:left_bound_eps]))
    
    R = E_refl_max**2 / E_inc_max**2
    
    print("Reflection coefficient R = {0}".format(R))
