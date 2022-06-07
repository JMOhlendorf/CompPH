# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 15:32:28 2020

@author: Christian Gorjaew, Julius Meyer-Ohlendorf
"""
import numpy as np
import matplotlib.pyplot as plt

def R_n_j(R, n, j):
    """
    Richards extrapolation recursion formula for j > 0
    """
    return R[n,j-1] + (R[n,j-1] - R[n-1,j-1]) / (4**j - 1)

def R_n_0(func, a, R, n, h_n, n_min):
    """
    Calculates R_n,0 from R_n-1,0 by adding the additional functionvalues for 
    n >= 1.
    """
    x = np.array([a + (2 * i - 1) * h_n for i in range(1, 2**(n + n_min - 1) + 1)])
    return 0.5 * R[n-1,0] + h_n * func(x).sum()

def trapz(func, a, b, n_min):
    if n_min == 0:
        return (func(a) + func(b)) * (b-a) / 2
    else:
        N = 2**n_min + 1
        x = np.linspace(a, b, N)
        h = x[1] - x[0]
        return 0.5 * h * (func(a) + func(b) + 2 * func(x[1:-1]).sum())

def romberg(func, a, b, n_max, n_min=0):
    """
    Calculates an integral using the Romberg scheme.

    Parameters
    ----------
    func : callable
        Input function f(x) to integrate.
    a : float
        Lower integration boundary.
    b : float
        Upper integration boundary.
    n_max : int
        Maximum n.
    n_min : int, optional
        Minimum value n. Altering n_min from 0 lets the Romberg scheme begin 
        at finer spacing h = (b - a) / 2^n_min. The default is 0.

    Returns
    -------
    array
        R_ii values for i=0..n_max.

    """
    R = np.zeros((n_max - n_min + 1, n_max - n_min + 1))  # Matrix containing R_n,j
    R[0,0] = trapz(func, a, b, n_min)  # Initial value
    
    # looping over all indics n and j to calulate all remaining R_n,j
    for n in range(1, n_max - n_min + 1):
        h_n = (b - a) / 2**(n + n_min)
        for j in range(n + 1):
            if j != 0:
                R[n,j] = R_n_j(R, n, j)
            else:
                R[n,j] = R_n_0(func, a, R, n, h_n, n_min)     
    
    return np.array([R[i,i] for i in range(0, n_max - n_min + 1)])
    
#%%
n_max = 29
i = np.arange(n_max + 1)
err = (1. / 2**i[0:7])**(2*i[0:7]+2)

# Integral (i)
exp_exact = np.exp(1) - 1
exp_rom = romberg(np.exp, 0, 1, n_max, 0)
delta_exp = exp_exact - exp_rom

# Integral (ii)
def sin_p(x):
    return np.sin(8 * x)**4
sin_exact = 3 * np.pi / 4
sin_rom = romberg(sin_p, 0, 2 * np.pi, n_max, 0)
delta_sin =  sin_exact - sin_rom

# Integral (iii)
sqrt_exact = 2 / 3
sqrt_rom = romberg(np.sqrt, 0, 1, n_max, 0)
delta_sqrt = sqrt_exact - sqrt_rom

# Integral (iii) for x=0.5..1.5
sqrt_exact_shift = 2 / 3 * (np.power(1.5, 1.5) - np.power(0.5, 1.5))
sqrt_rom_shift = romberg(np.sqrt, 0.5, 1.5, n_max, 0)
delta_sqrt_shift = sqrt_exact_shift - sqrt_rom_shift

#%% Plotting
plt.figure(figsize=(12,8))
plt.plot(i, np.abs(delta_exp), color="b", label="$f(x) = e^x$")
plt.plot(i[0:7], err, linestyle="-.", color="b",
         label="Theoretical error behaviour $\propto h_i^{2i+2}$ \n for $h_0 = 1$ ")

plt.plot(i, np.abs(delta_sin), color="r", label="$f(x) = \sin^4(8x)$")

plt.plot(i, np.abs(delta_sqrt), color="orange", label="$f(x) = \sqrt{x}$")
plt.plot(i, np.abs(delta_sqrt_shift), color="orange", 
         linestyle="--", label="$f(x) = \sqrt{x}$ for $x=0.5..1.5$")
         

plt.xlabel("$i$", fontsize=16)
plt.xticks(ticks=np.arange(0, n_max+1, dtype=int))
plt.xlim(0)
plt.ylabel("$|S(f(x)) - R_{ii}|$", fontsize=16)
plt.yscale("log")
plt.legend(fontsize=14)
plt.tight_layout()
plt.show()