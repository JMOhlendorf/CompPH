# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 21:32:38 2020

@author: Christian Gorjaew, Julius Meyer-Ohlendorf
"""

import numpy as np
import matplotlib.pyplot as plt

from newton import newton_1D_sket

class rungekutta_newton_1D(newton_1D_sket):
    
    def __init__(self, func, t_0, t_f, tau, r_0, v_0, **kwargs):
        assert(np.abs(r_0) <= np.pi)
        super(rungekutta_newton_1D, self).__init__(func, t_0, t_f, 
                                                   tau, r_0, v_0)
        self._kwargs = kwargs
        
    def _step(self, t):
        tau = self._tau
        tau_2 = tau / 2
        r_old = self._r[-1]
        v_old = self._v[-1]

        k1 =  tau * v_old
        kp1 = tau * self._acc(r_old, v_old, t, **self._kwargs)
        
        k2 = tau * (v_old + kp1 / 2)
        kp2 = tau * (self._acc(r_old + k1 / 2, v_old + kp1 / 2,
                        t + tau_2, **self._kwargs))
        
        k3 = tau * (v_old + kp2 / 2)
        kp3 = tau * (self._acc(r_old + k2 / 2, v_old + kp2 / 2, 
                        t + tau_2, **self._kwargs))
        
        k4 = tau * (v_old + kp3)
        kp4 = tau * (self._acc(r_old + k3, v_old + kp3, 
                        t + tau, **self._kwargs))
        
        r_new = r_old + (k1 + 2 * (k2 + k3) + k4) / 6
        v_new = v_old + (kp1 + 2 * (kp2 + kp3) + kp4) / 6
        
        self._r = np.append(self._r, r_new)
        self._v = np.append(self._v, v_new)
        
        self._t += tau

#%%
"""Parameters"""
k = 1
gamma = 0.5
omega = 2 / 3
Q = [0.5, 0.9, 1.2]

T_0 = 2 * np.pi / omega
tau = T_0 / 200

r_0 = 1
v_0 = 0

N = 2**16

"""Problem 2"""
def acc_pendulum_driven(r, v, t, Q):
    return -k * np.sin(r) - gamma * v + Q * np.sin(omega * t)

def wrap_pi(r):
    """
    Function that wraps the values of r into the interval [-pi, pi).
    """
    return r - (np.floor((r + np.pi) / (2 * np.pi)) * 2 * np.pi)

pend_q0 = rungekutta_newton_1D(acc_pendulum_driven, 0, N*tau,
                              tau, r_0, v_0, Q=Q[0])
r_q0, v_q0, time_q0 = pend_q0.solve()

print("Results at t_20 = T_0/10 = {2:.3f}: r(t_20) = {0:.3f}   v(t_20) = {1:.3f}".format(r_q0[20], v_q0[20], time_q0[20]))
# >> Results at t_20 = T_0/10 = 0.942: r(t_20) = 0.737   v(t_20) = -0.450

pend_q1 = rungekutta_newton_1D(acc_pendulum_driven, 0, N*tau,
                              tau, r_0, v_0, Q=Q[1])
r_q1, v_q1, time_q1 = pend_q1.solve()

pend_q2 = rungekutta_newton_1D(acc_pendulum_driven, 0, N*tau,
                              tau, r_0, v_0, Q=Q[2])
r_q2, v_q2, time_q2 = pend_q2.solve()

#%% Plotting
results = [[r_q0, v_q0, time_q0], 
           [r_q1, v_q1, time_q1], 
           [r_q2, v_q2, time_q2]]

samples_max = 8000
for i, result in enumerate(results):
    r_tmp, v_tmp, t_tmp = wrap_pi(result[0]), result[1], result[2]
    fig, ax = plt.subplots(1,2, figsize=(10,4), gridspec_kw={'width_ratios': [2, 1]})
    fig.suptitle("$Q$ = {}".format(Q[i]), fontsize=18)
    ax[0].plot(t_tmp[0:samples_max], r_tmp[0:samples_max])
    ax[0].set_xlabel("$t$", fontsize=16)
    ax[0].set_ylabel("$r(t)$", fontsize=16)
    ax[0].set_xlim(0, t_tmp[samples_max])
    ax[0].set_title("", fontsize=16)
    
    ax[1].scatter(r_tmp[2000:samples_max], v_tmp[2000:samples_max], 
                  s=0.5, label="$2000 \\tau \leq t \leq 8000 \\tau$")
    ax[1].set_xlabel("$r(t)$", fontsize=16)
    ax[1].set_ylabel("$v(t)$", fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    fig.savefig("./p2_Q{}.pdf".format(Q[i]), filetype="pdf")

#%%
"""Problem 3"""
from scipy.fftpack import fft

n = int(N / 2)
nu_max = 1 / tau / 2  # Maximum frequency detectable by fft
nu = np.linspace(0, nu_max, n + 1)  # Array containing sampled frequencies

nu_0 = omega / 2 / np.pi  # frequency of driving force
ind_max_plot = int(np.floor(6 * nu_0 / nu[1])) # index at 6 * nu_0

for i, result in enumerate(results):
    r_tmp = result[0]
    f = np.abs(fft(r_tmp))**2
    
    peridiogram = f[0:n+1]
    peridiogram[1:-1] += f[-1:-n:-1]
    peridiogram /= N**2
    
    plt.figure(figsize=(12,8))
    plt.semilogy(nu[0:ind_max_plot+1]/nu_0, peridiogram[0:ind_max_plot+1])
    plt.xlabel("$\\nu / \\nu_0$", fontsize=16)
    plt.ylabel("$P_l(\\nu)$", fontsize=16)
    plt.tight_layout()
    plt.savefig("./p3_Q{}.pdf".format(Q[i]), filetype="pdf")
    plt.close()
