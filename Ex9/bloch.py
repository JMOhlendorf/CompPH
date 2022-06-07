import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import h5py as h5


def calc_B(t, h, omega_0, B_0, phi):
    """
    Calculating the magnetic field at time t
    """

    Bx = h * np.cos(omega_0 * t + phi)
    By = -h * np.sin(omega_0 * t + phi)
    Bz = B_0

    return(Bx, By, Bz)

def product_step(M, t, inv_T2, inv_T1, tau, gamma, phi, h, omega_0, B_0):
    """
    Performing one product algorithm step on given M
    """

    # calculating matrix exponential of B
    e_tB = np.zeros((3,3))
    Bx, By, Bz = calc_B(t + tau / 2, h, omega_0, B_0, phi)

    Omega_2 = Bx**2 + By**2 + Bz**2
    Omega = np.sqrt(Omega_2)
    cos_Ot = np.cos(Omega * tau * gamma)
    sin_Ot = np.sin(Omega * tau * gamma)

    e_tB[0,0] = (Bx**2 + (By**2 + Bz**2) * cos_Ot) / Omega_2
    e_tB[0,1] = (Bx * By * (1 - cos_Ot) + Omega * Bz * sin_Ot) / Omega_2
    e_tB[0,2] = (Bx * Bz * (1 - cos_Ot) - Omega * By * sin_Ot) / Omega_2
    e_tB[1,0] = (Bx * By * (1 - cos_Ot) - Omega * Bz * sin_Ot) / Omega_2
    e_tB[1,1] = (By**2 + (Bx**2 + Bz**2) * cos_Ot) / Omega_2
    e_tB[1,2] = (By * Bz * (1 - cos_Ot) + Omega * Bx * sin_Ot) / Omega_2
    e_tB[2,0] = (Bx * Bz * (1 - cos_Ot) + Omega * By * sin_Ot) / Omega_2
    e_tB[2,1] = (By * Bz * (1 - cos_Ot) - Omega * Bx * sin_Ot) / Omega_2
    e_tB[2,2] = ( Bz**2 + (Bx**2 + By**2) * cos_Ot) / Omega_2

    # calculating matrix exponential of C
    e_tC2_diag = np.array([np.exp(-tau / 2 * inv_T2), np.exp(-tau / 2 * inv_T2), np.exp(-tau / 2 * inv_T1)])
    e_tC2 = np.diag(e_tC2_diag, 0)

    # final matrix exponential including C
    e_tBfinal = e_tC2 @ e_tB @ e_tC2

    M_new = e_tBfinal @ M

    return(M_new)

# simulation parameters
f_0 = 4
B_0 = 2 * np.pi * f_0
f_1 = 1 / 4
h = 2 * np.pi * f_1
gamma = 1
omega_0 = gamma * B_0

###### Change parameters here ####
time_max = 6
tau = 0.0025
m = int(time_max / tau) # number of product steps
time_arr = tau * np.arange(0, m+1)

simulating = True # whether new simulation should be performed
plotting = True # whether results of simulation should be plotted

# which case to simulate
case1 = True
case2 = False
case3 = False
case4 = False

inv_T1_arr = np.array([0, 0, 1, 1])
inv_T2_arr = np.array([0, 1, 0, 1])

if case1:
    M_initial = np.array([0, 1, 0])
    phi = 0
    extra_string = 'case1'

if case2:
    M_initial = np.array([1, 0, 0])
    phi = np.pi / 2
    extra_string = 'case2'

if case3:
    M_initial = np.array([0, 0, 1])
    phi = np.pi / 2
    extra_string = 'case3'

if case4:
    M_initial = np.array([1, 0, 1])
    phi = np.pi / 2
    extra_string = 'case4'

M = M_initial

# excecuting the actual simulation
##################################

def simulation(M, inv_T1, inv_T2, path):
    """
    Performing simulaton loop and saving generated results
    """

    # setting up storage frame
    storage_steps = m + 1

    with h5.File(path, "a") as file:
        # setting up the storage file of type 'hdf5'
        file.create_dataset('t', shape=(storage_steps,), maxshape=(None,),
                            chunks=True, dtype=float)
        file.create_dataset('Mx', shape=(storage_steps,), maxshape=(None,),
                            chunks=True, dtype=float)
        file.create_dataset('My', shape=(storage_steps,), maxshape=(None,),
                            chunks=True, dtype=float)
        file.create_dataset('Mz', shape=(storage_steps,), maxshape=(None,),
                            chunks=True, dtype=float)

        # actual simulation loop
        for step in range(0, m+1):

            # perform a product step
            M = product_step(M, time_arr[step], inv_T2, inv_T1, tau, gamma, phi, h, omega_0, B_0)

            file['t'][step] = time_arr[step]
            file['Mx'][step] = M[0]
            file['My'][step] = M[1]
            file['Mz'][step] = M[2]



if simulating:
    for i in range(len(inv_T1_arr)):
        print('Simulating loop:', str(i))
        # storage path
        path = './{0}_invT1_{1}_invT2_{2}.hdf5'.format(extra_string,
               inv_T1_arr[i], inv_T2_arr[i])
        simulation(M, inv_T1_arr[i], inv_T2_arr[i], path)


# creating plots of simulation
##################################

def plot_func(inv_T1, inv_T2, path):
    """
    Generates plot of Mx, My and Mz as function of time
    """

    data = h5.File(path, "r")

    t_arr = data['t'][:]
    Mx = data['Mx'][:]
    My = data['My'][:]
    Mz = data['Mz'][:]
    data.close()

    colors = ['b', 'g', 'r']

    fig, ax = plt.subplots(figsize=(4,3.5))
    title1 = '$\dfrac{{1}}{{T_1}}=${0},  $\dfrac{{1}}{{T_2}}=${1},  '.format(
              inv_T1, inv_T2)
    title2 = '$M(0)=${0}'.format(M_initial)
    title = title1 + title2
    ax.set_title(title, fontsize=12)
    ax.plot(t_arr, Mx, color=colors[0], label='$Mx(t)$')
    ax.plot(t_arr, My, color=colors[1], label='$My(t)$')
    ax.plot(t_arr, Mz, color=colors[2], label='$Mz(t)$')
    ax.set_xlim(0,t_arr[-1])
    if case3:
        ax.set_ylim(-1.1 * np.max(Mz), 1.1 * np.max(Mz))
    else:
        ax.set_ylim(-1.1 * np.max(Mx), 1.1 * np.max(Mx))
    ax.set_xlabel('$t$', fontsize=12)
    ax.set_ylabel('$M$', fontsize=12)
    ax.grid(linestyle='--', alpha=0.5)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig('{0}_invT1_{1}_invT2_{2}.pdf'.format(extra_string,
                inv_T1, inv_T2))



if plotting:
    for i in range(len(inv_T1_arr)):
        # storage path
        path = './{0}_invT1_{1}_invT2_{2}.hdf5'.format(extra_string,
               inv_T1_arr[i], inv_T2_arr[i])
        plot_func(inv_T1_arr[i], inv_T2_arr[i], path)
