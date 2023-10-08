from typing import Tuple

import numpy as np


def delta_phi(ticks: int, prev_ticks: int, resolution: int) -> Tuple[float, float]:
    """
    Args:
        ticks: Current tick count from the encoders.
        prev_ticks: Previous tick count from the encoders.
        resolution: Number of ticks per full wheel rotation returned by the encoder.
    Return:
        rotation_wheel: Rotation of the wheel in radians.
        ticks: current number of ticks.
    """

    alpha = 2 * np.pi / resolution
    delta_ticks = ticks - prev_ticks
    dphi = delta_ticks * alpha


    # TODO: these are random values, you have to implement your own solution in here
    #ticks = prev_ticks + int(np.random.uniform(0, 10))
    #dphi = np.random.random()
    # ---
    return dphi, ticks


def pose_estimation(
    R: float,
    baseline: float,
    x_prev: float,
    y_prev: float,
    theta_prev: float,
    delta_phi_left: float,
    delta_phi_right: float,
) -> Tuple[float, float, float]:

    """
    Calculate the current Duckiebot pose using the dead-reckoning model.

    Args:
        R:                  radius of wheel (both wheels are assumed to have the same size) - this is fixed in simulation,
                            and will be imported from your saved calibration for the real robot
        baseline:           distance from wheel to wheel; 2L of the theory
        x_prev:             previous x estimate - assume given
        y_prev:             previous y estimate - assume given
        theta_prev:         previous orientation estimate - assume given
        delta_phi_left:     left wheel rotation (rad)
        delta_phi_right:    right wheel rotation (rad)

    Return:
        x:                  estimated x coordinate
        y:                  estimated y coordinate
        theta:              estimated heading
    """

    # Distance travelled by each wheel
    dl = R * delta_phi_left
    dr = R * delta_phi_right
    # Distance travelled by the robot
    dA = (dr + dl)/2
    dTheta = (dr - dl) / baseline
    # Robot motion
    dx = dA * np.cos(theta_prev)
    dy = dA * np.sin(theta_prev)
    # Current pose estimation
    x_curr =     x_prev + dx
    y_curr =     y_prev + dy
    theta_curr = theta_prev + dTheta
    

    # These are random values, replace with your own
    # x_curr = np.random.random()
    # y_curr = np.random.random()
    # theta_curr = np.random.random()
    # ---
    return x_curr, y_curr, theta_curr
