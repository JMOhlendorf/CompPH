# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 21:22:16 2020

@author: cgorj
"""

import numpy as np

class newton_1D_sket(object):
    """
    Skeleton class for algorithms that solve Newton's equation in 1D.
    Parent of classes that implement specific solving methods. Those classes
    must provide a function 'self._step' implementing a single time step
    of the method in use.
    
    Parameters
    ----------
    func : callable
        Function describing the acceleration.
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
        self._n_points = int((t_f - t_0) / tau) + 1
        self._t0 = float(t_0)
        self._tf = float(t_f)
        self._tau = float(tau)
        self._r = np.array([r_0], dtype=float)
        self._v = np.array([v_0], dtype=float)
        self._t = t_0 + tau
    
    @property
    def tf(self):
        return self._tf
    
    @tf.setter
    def tf(self, tf_new):
        assert(tf_new > self._tf)
        self._tf = tf_new
        self._n_points = int((self._tf - self._t0) / self._tau) + 1
        
    def _step(self, t):
        pass
    
    def solve(self):
        while self._t <= self._tf:
            self._step(self._t)
        
        time = np.array([self._t0 + j * self._tau for j in np.arange(self._r.shape[0])])
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
    