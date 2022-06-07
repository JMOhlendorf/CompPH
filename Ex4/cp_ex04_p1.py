"""
@author: Julius Meyer-Ohlendorf, Christian Gorjaew
"""

import numpy as np
from numpy import cos, pi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def Jacobi(Phi, threshold, Niter_max = None):
    '''
    Relaxes Phi grid according to Jacobi scheme

    :param Phi: unrelaxed Phi grid
    :type Phi: array_like
    :param threshold: desired deviation between step i and i+1
    :type threshold: : float
    :param Niter_max: specifies numbers of iterations (optional)
    :type Niter_max: int or None

    :returns: relaxed grid and used iterations Niter
    :rtype: tuple
    '''
    Phi_new = np.copy(Phi)
    dPhi = 1
    Niter = 0
    while dPhi > threshold:
        if Niter_max != None and Niter >= Niter_max:
            break
        Niter += 1
        if Niter % 100 == 0:
            print('Niter:', Niter)
        for i in range(1, Phi.shape[0]-1):
            for j in range(1, Phi.shape[1]-1):
                Phi_new[i,j] = 1/4*(Phi[i,j-1] + Phi[i,j+1] +
                                    Phi[i-1,j] + Phi[i+1,j])

        dPhi = np.max(np.abs(Phi_new - Phi))
        if Niter % 100 == 0:
            print('dPhi:', dPhi)
        Phi, Phi_new = Phi_new, Phi

    return(Phi, Niter)

def GaussSeidel(Phi, threshold, Niter_max = None):
    '''
    Relaxes Phi grid according to GaussSeidel scheme

    :param Phi: unrelaxed Phi grid
    :type Phi: array_like
    :param threshold: desired deviation between step i and i+1
    :type threshold: : float
    :param Niter_max: specifies numbers of iterations (optional)
    :type Niter_max: int or None

    :returns: relaxed grid and used iterations Niter
    :rtype: tuple
    '''
    Phi_temp = np.copy(Phi)
    dPhi = 1
    Niter = 0
    while dPhi > threshold:
        if Niter_max != None and Niter >= Niter_max:
            break
        Niter += 1
        if Niter % 50 == 0:
            print('Niter:', Niter)
        for i in range(1, Phi.shape[0]-1):
            for j in range(1, Phi.shape[1]-1):
                Phi[i,j] = 1/4*(Phi[i,j-1] + Phi[i,j+1] +
                                Phi[i-1,j] + Phi[i+1,j])
        dPhi = np.max(np.abs(Phi - Phi_temp))
        if Niter % 50 == 0:
            print('dPhi:', dPhi)
        Phi_temp = np.copy(Phi)

    return(Phi, Niter)

def SOR(Phi, threshold, Niter_max = None):
    '''
    Relaxes Phi grid according to the Overrelaxation scheme including
    Chebyshev acceleration

    :param Phi: unrelaxed Phi grid
    :type Phi: array_like
    :param threshold: desired deviation between step i and i+1
    :type threshold: : float
    :param Niter_max: specifies numbers of iterations (optional)
    :type Niter_max: int or None

    :returns: relaxed grid and used iterations Niter
    :rtype: tuple
    '''
    Phi_temp = np.copy(Phi)
    Nsites = Phi.shape[0]*Phi.shape[1]
    omega = 1
    dPhi = 1
    Niter = 0
    while dPhi > threshold:
        if Niter_max != None and Niter >= Niter_max:
            break
        Niter += 1
        if Niter % 50 == 0:
            print('Niter:', Niter)

        # half sweep for "black" sites
        for i in range(1, Phi.shape[0]-1):
            for j in range(1, Phi.shape[1]-1):
                if (i+j) % 2 == 0:
                    Phi[i,j] = (1-omega)*Phi[i,j] + omega*1/4*(Phi[i,j-1] +
                               Phi[i,j+1] + Phi[i-1,j] + Phi[i+1,j])

        omega = 1/(1-omega/4*(1-pi**2/Nsites))

        # half sweep for "white" sites
        for i in range(1, Phi.shape[0]-1):
            for j in range(1, Phi.shape[1]-1):
                if (i+j) % 2 != 0:
                    Phi[i,j] = (1-omega)*Phi[i,j] + omega*1/4*(Phi[i,j-1] +
                               Phi[i,j+1] + Phi[i-1,j] + Phi[i+1,j])

        omega = 1/(1-omega/4*(1-pi**2/Nsites))
        dPhi = np.max(np.abs(Phi - Phi_temp))
        if Niter % 50 == 0:
            print('dPhi:', dPhi)
        Phi_temp = np.copy(Phi)

    return(Phi, Niter)

# setting up the discretized field Phi
Nx = 81
Ny = 81
Phi = np.zeros((Ny, Nx))


# filling Phi grid according to Dirichlet boundary conditions
for i in range(Nx):
    sx = pi/(Nx-1)
    Phi[0,i] = cos(-pi/2 + i*sx)
    Phi[Ny-1,:] = Phi[0,:]

for i in range(Ny):
    sy = pi/(Ny-1)
    Phi[i,0] = cos(pi/2 - i*sy)
    Phi[:,Nx-1] = Phi[:,0]


# setting up grids for plotting
x = np.linspace(-pi/2, pi/2, num=Nx, endpoint=True)
y = np.linspace(-pi/2, pi/2, num=Ny, endpoint=True)
X, Y =  np.meshgrid(x,y)


# choose: relaxation scheme, threshold and desired number of iterations (Niter)
threshold = 10**(-5) # choose from any threshold
Niter = None  # choose from: None or any integer
scheme = 'Jacobi'  # choose from: 'Jacobi', 'GaussSeidel', 'SOR'

if scheme == 'Jacobi':
    Phi, Niter = Jacobi(Phi, threshold, Niter_max = Niter)
    print('Niter for scheme:{0} and threshold={1}:'.format(scheme, threshold), Niter)

elif scheme == 'GaussSeidel':
    Phi, Niter = GaussSeidel(Phi, threshold, Niter_max = Niter)
    print('Niter for scheme:{0} and threshold={1}:'.format(scheme, threshold), Niter)

elif scheme == 'SOR':
    Phi, Niter = SOR(Phi, threshold, Niter_max = Niter)
    print('Niter for scheme:{0} and threshold={1}:'.format(scheme, threshold), Niter)

else:
    print('Something is wrong with the input parameters')

fig = plt.figure(0)
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Phi)
ax.set_xlabel('X', fontsize=16)
ax.set_ylabel('Y', fontsize=16)
ax.set_zlabel('$\Phi$', fontsize=16)
plt.title('Scheme:{0}   Niter={1}'.format(scheme, Niter), fontsize=16)
#plt.title('Initial $\Phi_{0}$ grid', fontsize=16)
fig.savefig('scheme{0}_Niter{1}.pdf'.format(scheme, Niter), bbox_inches='tight')
#fig.savefig('Initial.pdf', bbox_inches='tight')
plt.show()
