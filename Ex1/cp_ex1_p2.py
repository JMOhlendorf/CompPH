# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 16:52:28 2020

@author: Christian Gorjaew, Julius Meyer-Ohlendorf
"""
import numpy as np
from numpy import sin, pi
import matplotlib.pyplot as plt

def simpson(y, x):
    """
    Integrate 'y'('x') using Simpson's formula.

    Parameters
    ----------
    y : array_like
        Input array to integrate.
    x : array_like
        Sample points correspinding to 'y'.

    Returns
    -------
    S_S : float
        Integral value as approximized by Simpson's rule.

    """
    N = len(y)
    h = x[1] - x[0]
    
    # Collecting odd and even indices (except first and last/last two). Works 
    # for even and odd N, for example:
    # N = 9 or N = 10: index_even = [2,4,6], index_odd = [1,3,5,7]
    index_even = np.arange(2, N - 2, 2)
    index_odd = np.arange(1, N - 1, 2)
    
    S_S = (y[0] + 4 * np.sum(y[index_odd]) + 2 * np.sum(y[index_even]) + y[index_odd[-1] + 1]) * h / 3
    
    if not N % 2:
        S_S += (-y[-3] + 8 * y[-2] + 5 * y[-1]) * h / 12

    return S_S


#%% Evaluation for multiple values h
delta = []
h = []
integral_exact = 1.

for n in np.logspace(1, 8, 100, dtype=np.int64):
    
    x = np.linspace(0,pi / 2, n) 
    y = sin(x)
    h_tmp = x[1] - x[0]
    integral_simps = simpson(y, x)
    delta += [integral_exact - integral_simps]
    h += [h_tmp]
  
h = np.array(h)
delta = np.abs(np.array(delta))  # Absolute value for log-log plot
delta_theo = h**4 * (pi / 2 - 0) / 180  # Theoretical error divided by f^{4}(zeta)
#%% Plotting
plt.figure(figsize=(12,8))
plt.plot(h, delta, label="Calculated error $|\Delta_S| = |S - S_S|$")
plt.plot(h, delta_theo, color="r", label="Theoretical error $\Delta_S \\approx (b - a) \cdot h^4 / 180$")
plt.xlabel("$h$", fontsize=16)
plt.ylabel("$\Delta_S$", fontsize=16)
plt.gca().invert_xaxis()
plt.xscale("log")
plt.yscale("log")
plt.legend(fontsize=16)
plt.tight_layout()
plt.show()

