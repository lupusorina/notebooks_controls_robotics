from dataclasses import dataclass

import numpy as np
import math

def wrap_circular_value(input_value):
    return (input_value + np.pi) % (2*np.pi) - np.pi


@dataclass
class TwoDimDoubleIntegratorNominal:
    xdim: int = 4
    xnames = ["x", "y", "x_dot", "y_dot"]
    udim: int = 2
    unames = ["u1", "u2"]
    action_lims = lambda self: np.array([[-20.0, 20.0],
                                         [-20.0, 20.0]])

    def dynamics(self, state, action, dt):
        x, x_dot, y, y_dot = state
        u1, u2 = action

        x = x + dt * x_dot
        y = y + dt * y_dot

        next_state = np.array([x, x_dot, y, y_dot])

        Jx = np.eye(4) + dt * np.array([
            [0, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ])

        Ju = dt * np.array([
            [0, 0],
            [1, 0],
            [0, 0],
            [0, 1],
        ])

        return next_state, Jx, Ju


@dataclass
class TwoDimSingleIntegratorNominal:
    xdim: int = 2
    xnames = ["x", "y"]
    udim: int = 2
    unames = ["u1", "u2"]
    action_lims = lambda self: np.array([[-5.0, 5.0],
                                         [-5.0, 5.0]])

    def dynamics(self, state, action, dt):
        x, y = state

        u1, u2 = action

        x = x + dt * u1
        y = y + dt * u2
        next_state = np.array([x, y])

        # TODO: verify the Jacobians
        Jx = np.eye(2) + dt * np.zeros((2, 2))

        Ju = dt * np.array([
            [1.0, 0],
            [0, 1.0],
        ])

        return next_state, Jx, Ju


@dataclass
class Dubins:
    xdim: int = 3
    xnames = ["x", "y", "theta"]
    udim: int = 2
    unames = ["v", "omega"]

    def dynamics(self, state, action, dt):
        x, y, theta = state
        v, omega = action
        c = np.cos(theta)
        s = np.sin(theta)

        theta = theta + dt * omega
        x = x + dt * v * c
        y = y + dt * v * s

        next_state = np.array([x, y, theta])

        Jx = np.eye(3) + dt * np.array([
            [0, 0, -v*s],
            [0, 0,  v*c],
            [0, 0,    0],
        ])

        Ju = dt * np.array([
            [c, 0],
            [s, 0],
            [0, 1],
        ])

        return next_state, Jx, Ju

    def dubins_action(self, state, action, dt):
        return action


@dataclass
class Ackermann:
    xdim: int = 3
    xnames = ["x", "y", "theta"]
    udim: int = 2
    unames = ["v", "steering"]
    action_lims = lambda self: np.array([[-5.0, 5.0],
                                         [-np.deg2rad(28), np.deg2rad(28)]])

    def dynamics(self, state, action, dt):
        x, y, theta = state
        v, steering = action
        c = np.cos(theta)
        s = np.sin(theta)
        L = 1.0 # Length of the car.

        theta = theta + dt * v / L * np.tan(steering)
        theta = wrap_circular_value(theta)
        x = x + dt * v * c
        y = y + dt * v * s

        next_state = np.array([x, y, theta])

        Jx = np.eye(3) + dt * np.array([
            [0, 0, -v*s],
            [0, 0,  v*c],
            [0, 0,    0],
        ])

        Ju = dt * np.array([
                            [c,                         0],
                            [s,                         0],
                            [1/L*np.tan(steering), v/L/(np.cos(steering)**2)],
                            ])

        return next_state, Jx, Ju

    def ackermann_action(self, state, action, dt):
        return action


@dataclass
class AckermannVelDelay:
    xdim: int = 4
    xnames = ["x", "y", "theta", "vx"]
    udim: int = 2
    unames = ["u_v", "steering"]
    action_lims = lambda self: np.array([[-5.0, 5.0],
                                         [-np.deg2rad(28), np.deg2rad(28)]])

    def dynamics(self, state, action, dt):
        x, y, theta, vx = state
        u_v, steering = action
        c = np.cos(theta)
        s = np.sin(theta)
        L = 1.0 # Length of the car.
        tau = 0.1 # Time delay.

        theta = theta + dt * vx / L * np.tan(steering)
        theta = wrap_circular_value(theta)

        x = x + dt * vx * c
        y = y + dt * vx * s
        vx = vx + dt * (1/tau * (u_v - vx))

        next_state = np.array([x, y, theta, vx])

        Jx = np.eye(4) + dt * np.array([
            [0, 0, -vx*s, c],
            [0, 0,  vx*c, s],
            [0, 0,    0,  np.tan(steering)/L],
            [0, 0,    0,  -1/tau],
        ])

        Ju = dt * np.array([
                            [0,          0],
                            [0,          0],
                            [0.0,        vx/L/(np.cos(steering)**2)],
                            [1/tau,      0],
                            ])

        return next_state, Jx, Ju

    def ackermann_action(self, state, action, dt):
        return action


system_lookup = dict(
    dubins=Dubins,
    ackermann=Ackermann,
    ackermann_vel_delay=AckermannVelDelay,
    two_dim_double_integrator_nominal=TwoDimDoubleIntegratorNominal,
    two_dim_single_integrator_nominal=TwoDimSingleIntegratorNominal
)
