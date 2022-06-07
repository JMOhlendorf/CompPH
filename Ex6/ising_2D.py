import numpy as np
from numba import njit
import matplotlib.pyplot as plt

def calc_E(lattice, N):
    '''
    Calculates energy of 2D lattice

    :param lattice: lattice used to calculate energy
    :type lattice: ndarry shape (N+2, N+2), shape due to free BC
    :param N: number of spins
    :type N: : int

    :returns: calculated energy
    :rtype: float
    '''
    E = 0
    # sweeping through rows
    E -= np.sum(np.multiply(lattice[:,1:N], lattice[:,2:N+1]))
    # sweeping through columns
    E -= np.sum(np.multiply(lattice[1:N], lattice[2:N+1]))

    return(E)


@njit
def MMC_step(lattice, beta, N):
    '''
    Performs one step of the MMC algorithm on the lattice, so that resulting
    configurations are distributed according to the unknown distribution

    :param lattice: lattice used to perform one MMC step
    :type lattice: ndarry shape (N+2, N+2), shape due to free BC
    :param beta: inverse temperature
    :type beta: : float
    :param N: number of spins
    :type N: : int

    :returns: lattice
    :rtype: ndarray
    '''

    for k in range(N**2):
        # picking index of spin
        i = np.random.randint(1, N+1)
        j = np.random.randint(1, N+1)

        delta_E =  2 * lattice[i][j] * (lattice[i-1][j] + lattice[i+1][j] +
                                        lattice[i][j-1] + lattice[i][j+1])

        # deciding if flipped or not
        if delta_E < 0:
            lattice[i][j] *= -1

        else:
            q = np.exp(-beta * delta_E)
            r = np.random.uniform(0., 1.)
            if q > r:
                lattice[i][j] *= -1

    return(lattice)

# Paramters
######################################################
######################################################

# Parameters of the script
simulation = False
saving = False
evaluation = True
errorbar = True

# Parameters of the simulation
N_arr = np.array([10, 50, 100])
N_sample_arr = np.array([1000, 10000])
T_arr = np.arange(0.2, 4.2, 0.2)
beta_arr = 1 / T_arr

N_wait1 = 100000
N_wait2 = 200000

# Relaxation and measurement run
######################################################
######################################################
if simulation:

    for N in (N_arr):
        print('N:', str(N))

        for N_sample in (N_sample_arr):
            print('N_sample:', str(N_sample))

            for T in (T_arr):
                print('T:', str(T))
                beta = 1 / T

                # Initialization lattice
                # Due to free BC it is easier to add a ring of zeros around the
                # considered lattice for later calculations

                lattice = np.random.choice([1,-1], size=(N+2, N+2))
                lattice[:, 0] = 0
                lattice[:, N+1] = 0
                lattice[0, :] = 0
                lattice[N+1, :] = 0

                # Thermalization run:
                # Choosing N_wait depending on N and T
                if (N == 10) or (N == 50):
                    N_wait = N_wait1 if T < 2.6 else 10000
                else:
                    N_wait = N_wait2 if T < 2.6 else 10000

                for i in range(N_wait):
                    lattice = MMC_step(lattice, beta, N)

                # Measurement run:
                M_arr = np.zeros(N_sample)
                E_arr = np.zeros(N_sample)

                for i in range(N_sample):
                    lattice = MMC_step(lattice, beta, N)
                    M_arr[i] = np.sum(lattice[1:N+1, 1:N+1])
                    E_arr[i] = calc_E(lattice, N)

                if saving:
                    np.savez('./simulation/2D/N{0}_Nsample{1}_T{2:.1f}_{3}_{4}.npz'
                              .format(N, N_sample, T, N_wait1, N_wait2), E=E_arr, M=M_arr)


# Loading and plotting data from measurement
######################################################
######################################################
if evaluation:

    # three arrays holding mean and std of  U/N**2, C/N**2 and M/**2
    # for all parameters

    U_arr = np.zeros((len(N_arr), len(N_sample_arr), len(T_arr)))
    U_std_arr = np.zeros((len(N_arr), len(N_sample_arr), len(T_arr)))

    C_arr = np.zeros((len(N_arr), len(N_sample_arr), len(T_arr)))
    C_std_arr = np.zeros((len(N_arr), len(N_sample_arr), len(T_arr)))

    M_arr = np.zeros((len(N_arr), len(N_sample_arr), len(T_arr)))
    M_std_arr = np.zeros((len(N_arr), len(N_sample_arr), len(T_arr)))

    # theoretical values for M
    M_theo = np.zeros(len(T_arr))
    Tc = 2 / (np.log(1 + np.sqrt(2)))
    for i,T in enumerate(T_arr):
        if T < Tc:
            M_theo[i] = (1 - np.sinh(2 * beta_arr[i])**(-4))**(1/8)

    for j, N_sample in enumerate(N_sample_arr):

        fig_u, ax_u = plt.subplots(1)
        fig_c, ax_c = plt.subplots(1)
        fig_m, ax_m = plt.subplots(1)

        ax_m.plot(T_arr, M_theo, color='k', marker='>', label='Theoretical result')

        ax_u.set_xlabel("$T$", fontsize=14)
        ax_u.set_ylabel("$U / N^{2}$", fontsize=14)
        ax_u.set_title("$N_{sample}$ = " + "{}".format(N_sample), fontsize=14)
        ax_c.set_xlabel("$T$", fontsize=14)
        ax_c.set_ylabel("$C / N^{2}$", fontsize=14)
        ax_c.set_title("$N_{sample}$ = " + "{}".format(N_sample), fontsize=14)
        ax_m.set_xlabel("$T$", fontsize=14)
        ax_m.set_ylabel("$M / N^{2}$", fontsize=14)
        ax_m.set_title("$N_{sample}$ = " + "{}".format(N_sample), fontsize=14)

        colors = ["b", "g", "r"]
        marker = "."

        for i, N in enumerate(N_arr):

            for k, T in enumerate(T_arr):
                beta = 1 / T
                data = np.load('./simulation/2D/N{0}_Nsample{1}_T{2:.1f}_{3}_{4}.npz'
                                .format(N, N_sample, T, N_wait1, N_wait2))
                E = data['E'][:]
                M = data['M'][:]
                data.close()

                U = np.mean(E)
                U2 = U**2
                E2 = E**2
                E_std = np.std(E, ddof=1)

                # calculating mean and std
                U_arr[i,j,k] = U / N**2
                U_std_arr[i,j,k] = E_std / np.sqrt(N_sample) / N**2

                C_arr[i,j,k] = np.var(E, ddof=1) * beta**2 / N**2
                C_std_arr[i,j,k] = np.sqrt(2./(N_sample)) * np.var(E, ddof=1)
                C_std_arr[i,j,k] *= beta**2 / N**2

                M_arr[i,j,k] = np.abs(np.mean(M)) / N**2
                M_std_arr[i,j,k]  = np.std(M, ddof=1) / np.sqrt(N_sample) /N**2

            label = '$N_{spin} = $' + str(N)
            fmt = colors[i] + marker
            color = colors[i]

            if errorbar:
                ax_u.errorbar(T_arr, U_arr[i,j,:], yerr=U_std_arr[i,j,:],
                              fmt=fmt, label=label)
                ax_c.errorbar(T_arr, C_arr[i,j,:], yerr=C_std_arr[i,j,:],
                              fmt=fmt, label=label)
                ax_m.errorbar(T_arr, M_arr[i,j,:], yerr=M_std_arr[i,j,:],
                              fmt=fmt, label=label)
                ax_m.plot(T_arr, M_arr[i,j,:], color=color, marker=marker)
            else:
                ax_u.plot(T_arr, U_arr[i,j,:], color=color,
                          marker=marker, label=label)
                ax_c.plot(T_arr, C_arr[i,j,:], color=color,
                          marker=marker, label=label)
                ax_m.plot(T_arr, M_arr[i,j,:], color=color,
                          marker=marker, label=label)

        ax_u.axvline(x=Tc, color='k', label='$T_{C}$')
        ax_c.axvline(x=Tc, color='k', label='$T_{C}$')
        ax_m.axvline(x=Tc, color='k', label='$T_{C}$')
        ax_u.legend(fontsize=10)
        ax_u.grid(linestyle="--", alpha=0.5)
        ax_c.legend(fontsize=10)
        ax_c.grid(linestyle="--", alpha=0.5)
        ax_m.legend(fontsize=10)
        ax_m.grid(linestyle="--", alpha=0.5)
        fig_u.tight_layout()
        fig_c.tight_layout()
        fig_m.tight_layout()
        fig_u.savefig('./simulation/2D/U_errorbar_{0}_{1}_{2}_Nsample{3}.pdf'.format(str(errorbar), N_wait1, N_wait2, N_sample))
        fig_c.savefig('./simulation/2D/C_errorbar_{0}_{1}_{2}_Nsample{3}.pdf'.format(str(errorbar), N_wait1, N_wait2, N_sample))
        fig_m.savefig('./simulation/2D/M_errorbar_{0}_{1}_{2}_Nsample{3}.pdf'.format(str(errorbar), N_wait1, N_wait2, N_sample))
