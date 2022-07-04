import numpy as np

from system import System

class SingleTrack(System):
    """Single Track Vehicle

    state: [x, y, psi, sigma, gamma]

    x = vehicle CG x position
    y = vehicle CG y position
    psi = heading / yaw
    sigma = longitudinal speed
    gamma = steering angle

    inputs: [F, gamma_des, gamma_dot_des]

    F = driving force
    gamma_des = desired steering angle
    gamma_dot_des = desired steering rate

    equations of motion:

    x_dot = sigma * (cos(psi) - d / l * sin(psi) * tan(gamma))
    y_dot = sigma * (sin(psi) + d / l * cos(psi) * tan(gamma))
    psi_dot = sigma / l * tan(gamma)
    sigma_dot = (F - m_0 * tan(gamma) / cos^2(gamma) * gamma_dot * sigma) / (m + m_0 * tan^2(gamma))
    gamma_dot = 1 / tau_gamma * (gamma - gamma_des) + gamma_dot_des

    l = wheelbase
    d = distance from rear axle to CG
    m = mass
    J = inertia
    m_0 = effective mass = (m*d^2 + J) / l^2
    tau_gamma = steering time constant
    """
    
    n_states = 5
    m_inputs = 3

    def __init__(self, m, J, l, d, tau_gamma=0.1):

        self.m = m
        self.J = J
        self.l = l
        self.d = d
        self.tau_gamma = tau_gamma

        self.m_0 = (m*d**2 + J) / l**2

    def gamma_dot(self, gamma, gamma_des, gamma_dot_des):

        return 1 / self.tau_gamma * (gamma_des - gamma) + gamma_dot_des

    def f(self, x, u, t):

        x, y, psi, sigma, gamma = x
        F, gamma_des, gamma_dot_des = u

        gamma_dot = self.gamma_dot(gamma, gamma_des, gamma_dot_des)

        return np.array([
            sigma * (np.cos(psi) - self.d / self.l * np.sin(psi) * np.tan(gamma)),
            sigma * (np.sin(psi) + self.d / self.l * np.cos(psi) * np.tan(gamma)),
            sigma / self.l * np.tan(gamma),
            (F - self.m_0 * np.tan(gamma) / np.cos(gamma)**2 * gamma_dot * sigma) / (self.m + self.m_0 * np.tan(gamma)**2),
            gamma_dot,
        ])

class RearSteerSingleTrack(SingleTrack):
    """Rear Steer Single Track Vehicle

    equations of motion:

    x_dot = sigma * (cos(psi) + (l-d) / l * sin(psi) * tan(gamma))
    y_dot = sigma * (sin(psi) - (l-d) / l * cos(psi) * tan(gamma))
    psi_dot = sigma / l * tan(gamma)
    sigma_dot = (F - m_0 * tan(gamma) / cos^2(gamma) * gamma_dot * sigma) / (m + m_0 * tan^2(gamma))
    gamma_dot = 1 / tau_gamma * (gamma - gamma_des) + gamma_dot_des
    """

    def f(self, x, u, t):

        x, y, psi, sigma, gamma = x
        F, gamma_des, gamma_dot_des = u

        gamma_dot = self.gamma_dot(gamma, gamma_des, gamma_dot_des)

        return np.array([
            sigma * (np.cos(psi) + (self.l - self.d) / self.l * np.sin(psi) * np.tan(gamma)),
            sigma * (np.sin(psi) - (self.l - self.d) / self.l * np.cos(psi) * np.tan(gamma)),
            sigma / self.l * np.tan(gamma),
            (F - self.m_0 * np.tan(gamma) / np.cos(gamma)**2 * gamma_dot * sigma) / (self.m + self.m_0 * np.tan(gamma)**2),
            gamma_dot,
        ])

def _gen_limited_model(model):
    
    class LimitedModel(model):

        def __init__(self, m, J, l, d, sigma_max=np.inf, gamma_max=np.inf, gamma_dot_max=np.inf, F_accel_max=np.inf, F_decel_max=np.inf, **kwargs):

            super().__init__(m, J, l, d, **kwargs)

            self.sigma_max = sigma_max
            self.gamma_max = gamma_max
            self.gamma_dot_max = gamma_dot_max
            self.F_accel_max = F_accel_max
            self.F_decel_max = F_decel_max

        def gamma_dot(self, gamma, gamma_des, gamma_dot_des):

            gamma_dot = np.clip(1 / self.tau_gamma * (gamma_des - gamma) + gamma_dot_des, -self.gamma_dot_max, self.gamma_dot_max)

            if gamma > self.gamma_max:
                gamma_dot = np.clip(gamma_dot, None, 0)
            if gamma < -self.gamma_max:
                gamma_dot = np.clip(gamma_dot, 0, None)

            return gamma_dot

        def clip(self, x, u, t):

            x, y, psi, sigma, gamma = x
            F, gamma_des, gamma_dot_des = u

            gamma_des = np.clip(gamma_des, -self.gamma_max, self.gamma_max)

            gamma_dot = self.gamma_dot(gamma, gamma_des, gamma_dot_des)

            if sigma > self.sigma_max:
                F = np.clip(F, None, self.m_0 * np.tan(gamma) / np.cos(gamma)**2 * gamma_dot * self.sigma_max)
            
            if sigma < -self.sigma_max:
                F = np.clip(F, self.m_0 * np.tan(gamma) / np.cos(gamma)**2 * gamma_dot * (-self.sigma_max), None)
            
            F = np.clip(F, -self.F_decel_max, self.F_accel_max)

            x = np.array([x, y, psi, sigma, gamma])
            u = np.array([F, gamma_des, gamma_dot_des])

            return x, u, t

    return LimitedModel

LimitedSingleTrack = _gen_limited_model(SingleTrack)
LimitedRearSteerSingleTrack = _gen_limited_model(RearSteerSingleTrack)