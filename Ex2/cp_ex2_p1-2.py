# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 22:35:12 2020

@author: Christian Gorjaew, Julius Meyer-Ohlendorf
"""

import numpy as np
import matplotlib.pyplot as plt


class newton_1D_sket(object):
    """
    Skeleton class for algorithms that solve Newton's equation in 1D.
    Parent of classes that implement specific solving methods. Those classes
    must provide a function 'self._step' implementing a single time step
    of the method in use.
    
    Parameters
    ----------
    func : callable
        Function describing the acceleration. Signature func(r, t).
    t_0 : float
        Initial time.
    t_f : float
        Maximum time until which the equation should be solved.
    tau : float
        Time step.
    r_0 : float
        Initial value for the position.
    v_0 : float
        Initial value for the velocity
    """
    def __init__(self, func, t_0, t_f, tau, r_0, v_0):
        self._acc = func
        self._n_steps = int((t_f - t_0) / tau)
        self._t0 = t_0
        self._tf = t_f
        self._tau = tau
        self._r = np.array([r_0])
        self._v = np.array([v_0])
        self._t = t_0 + tau
    
    def _step(self, t):
        pass
    
    def solve(self):
        while self._t <= self._tf:
            self._step(self._t)
        
        time = np.arange(self._t0, self._tf, self._tau)
        return self._r, self._v, time
    

class euler(newton_1D_sket):
    """
    Implementation of Euler and Euler-Cromer methods.
    
    Parameters
    ----------
    Parameters of parent class
    
    cromer : bool
        If True, the Euler-Cromer method is used. Default is False resulting
        in use of regular Euler method.
    """
    def __init__(self, func, t_0, t_f, tau, r_0, v_0, cromer=False):
        super(euler, self).__init__(func, t_0, t_f, tau, r_0, v_0)
        self._cromer = int(not cromer)
    
    def _step(self,t):
        v_new = self._v[-1] + self._tau * self._acc(self._r[-1], t)
        self._v = np.append(self._v, v_new)

        r_new = self._r[-1] + self._tau * self._v[-1 - self._cromer]
        self._r = np.append(self._r, r_new)
        
        self._t += self._tau

        
class verlet_velocity(newton_1D_sket):
    """
    Implementation of Euler and Euler-Cromer methods.
    
    Parameters
    ----------
    Parameters of parent class
    
    """
    def __init__(self, func, t_0, t_f, tau, r_0, v_0):
        super(verlet_velocity, self).__init__(func, t_0, t_f, tau, r_0, v_0)
    
    def _step(self, t):
        acc = self._acc(self._r[-1], t)
        r_new = self._r[-1] + self._tau * self._v[-1] + 0.5 * self._tau**2 * acc
        
        t_new = t + self._tau
        acc_new = self._acc(r_new, t_new)
        v_new = self._v[-1] + 0.5 * self._tau * (acc + acc_new)
        
        self._r = np.append(self._r, r_new)
        self._v = np.append(self._v, v_new)
        
        self._t += self._tau
    
#%% Exercise 1
def acc_ho(r, t):
    return -r

t_0 = 0
t_f = 16 * np.pi
tau = 0.01
r_0 = 1
v_0 = 0
#%% (a) Euler method
eul = euler(acc_ho, t_0, t_f, tau, r_0, v_0, False)
eul_r, eul_v, eul_time = eul.solve()
eul_E = 0.5 * (eul_r**2 + eul_v**2)
#%% (b) Velocity Verlet method
verl = verlet_velocity(acc_ho, t_0, t_f, tau, r_0, v_0)
verl_r, verl_v, verl_time = verl.solve()
verl_E = 0.5 * (verl_r**2 + verl_v**2)
#%% (c) Euler-Cromer method
eul_cro = euler(acc_ho, t_0, t_f, tau, r_0, v_0, True)
eul_cro_r, eul_cro_v, eul_cro_time = eul_cro.solve()  
eul_cro_E = 0.5 * (eul_cro_r**2 + eul_cro_v**2)
#%%
# Results for r(t_n)
plt.figure(figsize=(8,8))
plt.plot(eul_time, eul_r, label="Euler")
plt.plot(verl_time, verl_r, label="Velocity Verlet")
plt.plot(eul_cro_time, eul_cro_r, linestyle="-.", label="Euler-Cromer")
plt.axhline(y=1, color="r", label="$r_{max}$ = 1")
plt.xlabel("$t_n$", fontsize=16)
plt.xlim(t_0,t_f)
plt.ylabel("$r(t_n)$", fontsize=16)
plt.legend(fontsize=14)
plt.tight_layout()
plt.show()
# Residuals
plt.figure(figsize=(8,8))
plt.plot(eul_time, eul_r-np.cos(eul_time), label="Euler")
plt.plot(verl_time, verl_r-np.cos(verl_time), label="Velocity Verlet")
plt.plot(eul_cro_time, eul_cro_r-np.cos(eul_cro_time), linestyle="-.", label="Euler-Cromer")
plt.xlabel("$t_n$", fontsize=16)
plt.xlim(t_0,t_f)
plt.ylabel("$r(t_n) - \cos (t_n)$", fontsize=16)
plt.legend(fontsize=14)
plt.tight_layout()
plt.show()
# Energy
plt.figure(figsize=(8,8))
plt.plot(eul_time, eul_E, label="Euler")
plt.plot(verl_time, verl_E, label="Velocity Verlet")
plt.plot(eul_cro_time, eul_cro_E, label="Euler-Cromer")
plt.axhline(y=eul_E[0], color="r", linestyle="dotted", label="Exact energy")
plt.xlabel("$t_n$", fontsize=16)
plt.ylabel("$E_n$", fontsize=16)
plt.xlim(t_0,t_f)
plt.legend(fontsize=14)
plt.tight_layout()
plt.show()
# Energy zoom
plt.figure(figsize=(8,8))
plt.plot(eul_time, eul_E, label="Euler")
plt.plot(verl_time, verl_E, label="Velocity Verlet")
plt.plot(eul_cro_time, eul_cro_E, label="Euler-Cromer")
plt.axhline(y=eul_E[0], color="r", linestyle="dotted", label="Exact energy")
plt.xlabel("$t_n$", fontsize=16)
plt.ylabel("$E_n$", fontsize=16)
plt.xlim(t_0,t_0 + 5e2 * tau)
plt.ylim(eul_E[0] - 0.0001, eul_E[0] + 0.0001)
plt.legend(loc=1, fontsize=14)
plt.tight_layout()
plt.show()
#%% Exercise 2
from scipy.fftpack import fft

def acc_pendulum(r, t):
    if np.abs(r) > np.pi:
        raise RuntimeWarning("r outside of allowed range")
    return -np.sin(r)

n_samples = 10  # Number of x_max samples
tau = 0.001
T = []  # container for period values
x_max_a = []  # container for x_max values

for i, x_max in enumerate(np.linspace(0.01,np.pi-0.1,n_samples)):
    print(i)
    verl_pend = verlet_velocity(acc_pendulum, t_0, t_f, tau, x_max, v_0)
    verl_pend_r, verl_pend_v, verl_pend_time = verl_pend.solve()
    Fr = fft(verl_pend_r)
    N_points = verl_pend_r.shape[0]
    Fr_pos = np.abs(Fr[0:N_points//2])
    freq = np.linspace(0, 1 / (2 * tau), N_points // 2)
    argmax = np.argmax(Fr_pos)
    T += [1 / freq[argmax]]
    x_max_a += [x_max]
    print(1/freq[argmax])
    
T = np.array(T)
x_max_a = np.array(x_max_a)
#%% Plotting
plt.figure(figsize=(12,8))
plt.plot(x_max_a, T, label="Ideal pendulum")
T_harm = 2 * np.pi  # Period of harmonic oscillator with omega = 1
plt.axhline(y=T_harm, color="r", linestyle="--", label="Harmonic approximation")
plt.xlabel("$x_{max}$", fontsize=16)
plt.ylabel("period $T(x_{max})$", fontsize=16)
plt.xlim(x_max_a[0], x_max_a[-1])
plt.legend(fontsize=16)
plt.tight_layout()
plt.show()