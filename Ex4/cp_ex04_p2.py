"""
Created on Mon May  4 19:50:09 2020
Python 3.7

@author: Christian Gorjaew, Julius Meyer-Ohlendorf
"""

import numpy as np
from numpy.linalg import norm

def SD(A, b, x_0, eps=1e-10):
    """
    Steepest descent minimization method for quadratic function 
    f(r) = r * Ar + b * r

    Parameters
    ----------
    A : 2darray
        Hessian matrix A
    b : 1darray
        Vector b
    x_0 : 1darray
        Starting point for r
    eps : float, optional
        Threshold value. The default is 1e-10.

    Returns
    -------
    r_k : 1darray
        Approximated minimum of function f(r).
    n_k_norm : float
        Euclidian norm of last direction vector n_k.
    it : int
        Number of itreation at which threshold is reached.

    """
    n_k = -A @ x_0 + b  # "@" perform multiplication btw A and x_0 accrding to matrix multipliaction rules
    r_k = x_0.copy()
    it = 0
    n_k_norm = norm(n_k)
    
    while n_k_norm > eps:
        lam_k = n_k_norm**2 / (n_k @ (A @ n_k))
        r_k += lam_k * n_k
        n_k = -A @ r_k + b
        it += 1
        n_k_norm = norm(n_k)

    return r_k, n_k_norm, it

def CG(A, b, x_0, eps=1e-10, max_iter=None, multA=None):
    """
    Conjugate-gradient method for quadratic function 
    f(r) = r * Ar + b * r using Hestenes-Stiefel scheme.

    Parameters
    ----------
    A : 2darray
        Hessian matrix A. If multA is given, set to None.
    b : 1darray
        Vector b
    x_0 : 1darray
        Starting point for r
    eps : float, optional
        Threshold value. The default is 1e-10.
    max_iter : int, optional
        Maximum number of executed iteration. If not given, max_iter is set to
        2 * b.shape[0]
    multA : callable, optional
        If given, matrix multiplication is performed using the given function
        (e.g., if the matrix is sparse).
        If not given, conventional matrix multiplication is used (using  "@")

    Returns
    -------
    r_k : 1darray
        Approximated minimum of function f(r).
    n_k_norm : float
        Euclidian norm of last direction vector n_k.
    it : int
        Number of itreation at which threshold is reached.
    """
    
    if max_iter is None:
        max_iter = 2 * b.shape[0]
    
    if multA is None:
        assert(A.shape[0] == A.shape[1])
        
        def multA(x):
            return A @ x

    
    n_k = -multA(x_0) + b
    g_k = n_k.copy()
    r_k = x_0.copy()
    it = 0
    n_k_norm = norm(n_k)
    
    while n_k_norm > eps and it < max_iter:
        A_n_k = multA(n_k)
        g_k_scalar = g_k @ g_k
        
        lam_k = g_k_scalar / (n_k @ (A_n_k))
        r_k += lam_k * n_k
        g_k -= lam_k * A_n_k
        
        n_k *= (g_k @ g_k) / g_k_scalar
        n_k += g_k
        
        n_k_norm = norm(n_k)
        it += 1
    
    return r_k, n_k_norm, it

#%% Problem 2 a) %%#

A = np.loadtxt("./CG_Matrix_10x10.dat", dtype="f8", delimiter=" ")  
x_0 = np.ones(A.shape[1])
b = x_0

r_SD, n_norm_SD, n_it_SD = SD(A, x_0, b, eps=1e-10)
print("SD: x[0] = {0}, \t |x| = {1}, \t n_it = {2}".format(r_SD[0], norm(r_SD), n_it_SD))
# >>> SD: x[0] = 43.883667667433954, 	 |x| = 157.79891310794955, 	 n_it = 95072

r_CG, n_norm_CG, n_it_CG = CG(A, x_0, b, eps=1e-10)
print("CG: x[0] = {0}, \t |x| = {1}, \t n_it = {2}".format(r_CG[0], norm(r_CG), n_it_CG))
# >>> CG: x[0] = 43.88366767134392, 	 |x| = 157.7989131224579, 	 n_it = 12

#%% Problem 2 b) %%#

from numpy import cos, pi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def multA(x):
    """
    Multiplies sparse matrix describing the discretized Poisson equation with
    a vector x.
    
    Assumes a 2D grid and a quadratic marix.

    Parameters
    ----------
    x : array
        1D-array corresponding to the function values of Phi_i,j of form
        x = [Phi_1,1, Phi_1,2, ... , Phi_1,N-1, Phi_2,1, ..., Phi_N-1,N-1]

    Returns
    -------
    x_out : array
        Result of the matrix vector mutiplication x_out = A * x

    """
    x_out = np.zeros_like(x)
    N_p = x.shape[0]
    N = int(np.sqrt(N_p))  # assuming square matrix thus, N_p = N * N for now
    n = N
    m = N

    for j in range(n):
        for k in range(m):
            i = j * m + k

            if j == 0:  # left block of matrix A corresponding to sites adjacent to bottom grid boundary
                if k % m == 0:  # left edge within block corresponding to sites adjacent to  left grid boundary
                    x_out[i] = 4 * x[i] - x[i+1] - x[i+m]
                elif k % m == (m - 1):  # right edge within block corresponding to sites adjacent to right grid boundary
                    x_out[i] = 4 * x[i] - x[i-1] - x[i+m]
                else:  # sites not adjacent to left or right boundary
                    x_out[i] = 4 * x[i] - x[i+1] - x[i-1] - x[i+m]
            elif j == (n - 1): # Right block of matrix A corresponding to sites adjacent to top grid edge
                if k % m == 0:
                    x_out[i] = 4 * x[i] - x[i+1] - x[i-m]
                elif k % m == (m - 1):
                    x_out[i] = 4 * x[i] - x[i-1] - x[i-m]
                else:
                    x_out[i] = 4 * x[i] - x[i+1] - x[i-1] - x[i-m]
            else:  # central blocks correspondig to grid sites not adjacent to top or bottom edge
                if k % m == 0:
                    x_out[i] = 4 * x[i] - x[i+1] - x[i+m] - x[i-m]
                elif k % m == (m - 1):
                    x_out[i] = 4 * x[i] - x[i-1] - x[i+m] - x[i-m]
                else:
                    x_out[i] = 4 * x[i] - x[i+1] - x[i-1] - x[i+m] - x[i-m]
    
    return x_out


# setting up the discretized field Phi_grid
Nx = 81
Ny = 81
Phi_grid = np.zeros((Ny, Nx))


# filling Phi_grid grid according to Dirichlet boundary conditions
for i in range(Nx):
    sx = pi/(Nx-1)
    Phi_grid[0,i] = cos(-pi/2 + i*sx)
    Phi_grid[Ny-1,:] = Phi_grid[0,:]

for i in range(Ny):
    sy = pi/(Ny-1)
    Phi_grid[i,0] = cos(pi/2 - i*sy)
    Phi_grid[:,Nx-1] = Phi_grid[:,0]
    
# Constructing vector b_tissue consisting of boundary values only for Laplace eq.
Phi_grid_tmp = Phi_grid.copy()  # temporary grid to set up b_tissue
# loop adds boundary values to direct neighbors on variable grid
for i in range(1, Nx-1):
    if i == 1:
        Phi_grid_tmp[i, 1:-1] += Phi_grid_tmp[0, 1:-1]
    elif i == (Nx - 2):
        Phi_grid_tmp[i, 1:-1] += Phi_grid_tmp[i+1, 1:-1]
    
    Phi_grid_tmp[i, 1] += Phi_grid_tmp[i, 0]
    Phi_grid_tmp[i, -2] += Phi_grid_tmp[i, -1]

b_tissue = Phi_grid_tmp[1:-1,1:-1].flatten()  # vector b containing boundary condition
phi_init = np.zeros((Nx-2)*(Ny-2))  # start value for minimization
del Phi_grid_tmp

############### Checking number of iterations until eps < 10^-5 ###############
Phi, error_tissue, niter_tissue = CG(None, b_tissue, phi_init, 
                                     eps=1e-5, multA=multA)
print("Number of iterations until |n_k| < 10^-5: {}".format(niter_tissue))
# >>> Number of iterations until |n_k| < 10^-5: 54


####### Calculating Phi for different numbers of iterations + 3D plot ########
# setting up grids for plotting
x = np.linspace(-pi/2, pi/2, num=Nx, endpoint=True)
y = np.linspace(-pi/2, pi/2, num=Ny, endpoint=True)
X, Y =  np.meshgrid(x,y)

Niters = np.array([10, 50, 100])
acc = []
for Niter in Niters:
    print("Current number of max_iter: {}".format(Niter))
    Phi, error_tissue, niter_tissue = CG(None, b_tissue, phi_init, 
                                         max_iter=Niter, eps=0, multA=multA)
    
    # Folding vector Phi back onto grid
    Phi_grid_it = Phi_grid.copy()
    Phi_grid_it[1:-1,1:-1] = Phi.reshape((Nx-2,Ny-2))
    
    fs = 16
    fig = plt.figure(0)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Phi_grid_it)
    ax.set_xlabel('$x$', fontsize=fs)
    ax.set_ylabel('$y$', fontsize=fs)
    ax.set_zlabel('$\Phi$', fontsize=fs)
    plt.title('Scheme: CG  \t Niter={0} \t $|n_k|$ = {1:.2g}'.format(Niter, error_tissue), fontsize=fs)
    fig.savefig('CG_Niter{0}.pdf'.format(Niter), bbox_inches='tight')
    plt.show()
    
    acc += [error_tissue]
#%% Plots number of iterations vs. accuaracy. Adjust array "Niters" and 
#   uncomment
# acc = np.array(acc)
# plt.semilogy(Niters, acc, marker="o")
# plt.ylabel("$|n_{k=N}|$", fontsize=16)
# plt.xlabel("Iterations $N$", fontsize=16)
# plt.xlim(10,101)
# plt.xticks(np.arange(10,101,10))
# plt.tight_layout()
# plt.show()
    
