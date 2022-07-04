"""Single Track Model Test

Tests front wheel steer and rear wheel steer models with a sinusoidal steering input
"""

import numpy as np
from matplotlib import pyplot as plt

from sim import Sim
from single_track import SingleTrack, RearSteerSingleTrack

m = 1500 # kg
J = 2500 # kg
l = 2.5 # m
d = 1 # m

sys_f = SingleTrack(m, J, l, d)
sys_r = RearSteerSingleTrack(m, J, l, d)

gamma = lambda t: np.pi / 72 * np.sin(2 * np.pi * t) + np.pi/18
gamma_dot = lambda t: np.pi**2 / 36 * np.cos(2 * np.pi * t)

def u(x, t):

    x, y, psi, sigma, gamma = x

    return np.array([
        sys_f.m_0 * np.tan(gamma) / np.cos(gamma)**2 * gamma_dot(t) * sigma,
        gamma,
        gamma_dot(t),
    ])

t = np.linspace(0, 10, 1000)

sim_f = Sim(sys_f, np.array([0, 0, 0, 20, gamma(0)]))

sim_f.solve(u, t)

xs_f, us_f, ts_f = sim_f.get_result()
xdots_f = sys_f.f(xs_f.T, us_f.T, ts_f.T).T

sim_r = Sim(sys_r, np.array([0, 0, 0, 20, gamma(0)]))

sim_r.solve(u, t)

xs_r, us_r, ts_r = sim_r.get_result()
xdots_r = sys_f.f(xs_r.T, us_r.T, ts_r.T).T

plt.figure()
plt.plot(xs_f[:, 0], xs_f[:, 1])
plt.plot(xs_r[:, 0], xs_r[:, 1])
plt.xlabel('x')
plt.ylabel('y')

plt.figure()
plt.plot(ts_f, xs_f[:, 2])
plt.plot(ts_r, xs_r[:, 2])
plt.xlabel('t')
plt.ylabel('phi')

plt.figure()
plt.plot(ts_f, xs_f[:, 3])
plt.plot(ts_r, xs_r[:, 3])
plt.xlabel('t')
plt.ylabel('sigma')

plt.show()