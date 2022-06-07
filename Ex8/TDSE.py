import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import h5py as h5

# simulation parameters
sigma = 3.0
x_0 = 20.0
q = 1.0
Delta = 0.1
L = 1001
tau = 0.001
m = 50000

###### Change parameters here ####
barrier = False  # whether simulating with barier
simulation = False  # whether new simualtion data should be produced
animate = False # whether simulation data should be animated
plotting = False # whether simualtion data should be plotted
compare_plot = True # whether compare plot should be performed
P_norm_thres = 0.0001 # in case of a barrier, probability distribution
                     # is only normalized once probablity at grid point at
                     # the right of the barrier surpasses this value

######

x_arr = np.linspace(0., (L-1)*Delta, num=L)
time_arr = tau * np.arange(0,m+1)

# building potential in simulation box
barrier_xleft = 50.0
barrier_xright = 50.5
index_left = int(barrier_xleft / Delta)
index_right = int(barrier_xright / Delta)
V_arr = np.zeros(L)

if barrier:
    V_arr[index_left: index_right + 1] = 2
    extra_string = 'barrierNorm'
else:
    extra_string = 'NObarrier'

# calculateing approximate time-evolution operator U
V_diag_0 = np.exp(-1j * (tau * (1 + Delta**2 * V_arr) / Delta**2))
exp_V = np.diag(V_diag_0)

c = np.cos(tau  / (4 * Delta**2))
s = np.sin(tau  / (4 * Delta**2))

K1_diag_0 = np.full(L, c)
K1_diag_0[-1] = 1
K1_diag_1 = np.full(L-1, 1j * s)
K1_diag_1[1:L:2] = 0
exp_K1 = np.diag(K1_diag_1, -1) + np.diag(K1_diag_0, 0) + np.diag(K1_diag_1, 1)

K2_diag_0 = np.full(L, c)
K2_diag_0[0] = 1
K2_diag_1 = np.full(L-1, 1j * s)
K2_diag_1[0:L:2] = 0
exp_K2 = np.diag(K2_diag_1, -1) + np.diag(K2_diag_0, 0) + np.diag(K2_diag_1, 1)

U = (exp_K1 @ exp_K2 @ exp_V @ exp_K2 @ exp_K1)

# Initialize Phi
Phi = ((2 * np.pi * sigma**2)**(-1 / 4) * np.exp(1j * (x_arr - x_0)) *
        np.exp(-(x_arr - x_0)**2 / (4 * sigma**2)))


# setting up storage frame
storage_frame = 500
storage_steps = int(m / storage_frame) + 1

# storage path
path = './TDSE' + extra_string + '.hdf5'

# excecuting the actual simulation
##################################
if simulation:

    with h5.File(path, "a") as file:
        # setting up the storage file of type 'hdf5'
        file.create_dataset("t", shape=(storage_steps,), maxshape=(None,),
                            chunks=True, dtype=float)
        file.create_dataset("P", shape=(storage_steps, L), maxshape=(None, L),
                            chunks=True, dtype=float)

        k_storage = 0
        count = 0
        normalize = False

        # actual simulation loop
        for step in range(0, m+1):

            # application of approximate operator U
            Phi = U @ Phi

            if step % 2000 == 0:
                print('step:', str(step))

            if (step % storage_frame) == 0:
                # calculating probability
                P = np.absolute(Phi)**2 * Delta

                # Normalizing if x > 50.5
                # and P[index_right+1] > P_norm_thres

                if barrier:
                    P_barrier_right = P[index_right+1]

                    if (count == 0) and P_barrier_right > P_norm_thres:
                        normalize = True
                        count += 1

                    if normalize:
                        #print('I am normalizing:')
                        norm_fac = np.sum(P[index_right+1:])
                        #print('norm_fac', norm_fac)
                        P[index_right+1:] = P[index_right+1:] / norm_fac


                file["t"][k_storage] = time_arr[step]
                file["P"][k_storage] = P

                k_storage += 1


def animate_P(path, extra_string=""):
    """
    Generates a gif displaying the solution for the probability that was
    simulated with the product formula algorithm
    """
    fig, ax = plt.subplots(figsize=(5,3.5))
    fig.set_tight_layout(True)


    lineP, = ax.plot(x_arr, np.zeros_like(x_arr), linewidth=0.75)

    def update(i):
        title = 'time={0:.1f}'.format(t[i])
        lineP.set_ydata(P[i])
        ax.set_title(title, fontsize=12)

        return (lineP, ax)

    data = h5.File(path, "r")

    t = data["t"][:]
    P = data["P"][:]
    data.close()

    if barrier:
        ax.axvspan(x_arr[index_left], x_arr[index_right], color='green',
                   alpha=0.3, label='barrier')
    ax.set_xlim(0,x_arr[-1])
    ax.set_ylim(0, 0.014)
    ax.set_xlabel("$x$", fontsize=12)
    ax.set_ylabel("$P(x,t)$", fontsize=12)
    ax.grid(linestyle="--", alpha=0.5)
    ax.legend(loc=1, fontsize=10)

    anim = FuncAnimation(fig, update, frames=np.arange(t.shape[0]))

    anim.save('{0}.gif'.format(extra_string), writer=PillowWriter(fps=60))
    plt.show()

if animate:
    animate_P(path, extra_string)


def plot_func(path, extra_string=''):
    """
    Generates plot of probabilities at times t=0.0, 5.0, 40.0, 45.0, 50.0
    """

    data = h5.File(path, "r")

    t = data["t"][:]
    P = data["P"][:]
    data.close()

    times_plot = [0.0, 5.0, 40.0, 45.0, 50.0]
    colors = ['b', 'g', 'r', 'k', 'orange']

    fig, ax = plt.subplots(figsize=(6,3.5))
    if barrier:
        ax.axvspan(x_arr[index_left], x_arr[index_right], color='green',
                   alpha=0.3, label='barrier')
    ax.set_xlim(0,x_arr[-1])
    ax.set_ylim(0, 0.014)
    ax.set_xlabel('$x$', fontsize=12)
    ax.set_ylabel('$P(x,t)$', fontsize=12)
    ax.grid(linestyle='--', alpha=0.5)

    for i, times in enumerate(times_plot):

        index_plot = np.where(t == times)[0][0]
        ax.plot(x_arr, P[index_plot], color=colors[i],
                label='$t={0:.0f}$'.format(times))

    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig('{0}.pdf'.format(extra_string))

if plotting:
    plot_func(path, extra_string)

def compare_plot():
    """
    Generates plot of probabilities at times t=40.0, 45.0, 50.0 for both cases
    in order to visualize increased speed of tunneled wave packet
    """
    path_bar = './TDSEbarrierNorm.hdf5'
    path_NObar = './TDSENObarrier.hdf5'

    data_bar = h5.File(path_bar, "r")
    t_bar = data_bar["t"][:]
    P_bar = data_bar["P"][:]
    data_bar.close()

    data_NObar = h5.File(path_NObar, "r")
    t_NObar = data_NObar["t"][:]
    P_NObar = data_NObar["P"][:]
    data_NObar.close()

    times_plot = [40.0, 45.0, 50.0]
    colors = ['r', 'k', 'orange']

    fig, ax = plt.subplots(figsize=(6,3.5))
    ax.set_xlim(0,x_arr[-1])
    ax.set_ylim(0, 0.014)
    ax.set_xlabel('$x$', fontsize=12)
    ax.set_ylabel('$P(x,t)$', fontsize=12)
    ax.grid(linestyle='--', alpha=0.5)

    for i, times in enumerate(times_plot):

        index_plot_bar = np.where(t_bar == times)[0][0]
        index_plot_NObar = np.where(t_NObar == times)[0][0]
        ax.plot(x_arr, P_bar[index_plot_bar], color=colors[i],
                label='$t={0:.0f}$'.format(times))
        ax.plot(x_arr, P_NObar[index_plot_NObar],'--', color=colors[i],
                label='$t={0:.0f}$'.format(times))
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig('compare.pdf')

if compare_plot:
    compare_plot()
