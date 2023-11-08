from typing import Tuple

import numpy as np

# 480,640

def get_motor_left_matrix(shape: Tuple[int, int]) -> np.ndarray:
    # TODO: write your function instead of this one
    res = np.zeros(shape=shape, dtype="float32")
    # these are random values
    #res[:, 0:320] = -1
    #res[:, 320:] = 1
    # ---

    # First quadrant 1/9
    res[0:160, 0:] = .45       # Top
    res[160:, 0:213] = .25      # Left
    #res[160:320, 213:426] = .3  # Middle
    res[160:, 426:] = .2       # Right
    
    

    return res

"""
def get_motor_left_matrix(shape: Tuple[int, int]) -> np.ndarray:
    # Initialize matrix
    res = np.zeros(shape=shape, dtype="float32")
    # Divide image into 9 sections and tune actions accordingly
    for n in range(0, res.shape[0]):
        for m in range(0, res.shape[1]):
            # Top row (object farthest)
            if 0 <= n <= (res.shape[0] // 3) and 0 <= m <= (res.shape[1] // 3):
                res[n, m] = 0.45
            if 0 <= n <= (res.shape[0] // 3) and (res.shape[1] // 3) <= m <= (res.shape[1] * 2 // 3):
                res[n, m] = 0.40
            if 0 <= n <= (res.shape[0] // 3) and (res.shape[1] * 2 // 3) <= m <= res.shape[1]:
                res[n, m] = 0.40
            # Middle row
            if (res.shape[0] // 3) <= n <= (res.shape[0] * 2 // 3) and 0 <= m <= (res.shape[1] // 3):
                res[n, m] = 0.25
            if (res.shape[0] // 3) <= n <= (res.shape[0] * 2 // 3) and (res.shape[1] // 3) <= m <= (res.shape[1] * 2 // 3):
                res[n, m] = 0.25
            if (res.shape[0] // 3) <= n <= (res.shape[0] * 2 // 3) and (res.shape[1] * 2 // 3) <= m <= res.shape[1]:
                res[n, m] = 0.20
            # Bottom row (object closest)
            if (res.shape[0] // 3) <= n <= res.shape[0] and 0 <= m <= (res.shape[1] // 3):
                res[n, m] = 0.10
            if (res.shape[0] // 3) <= n <= res.shape[0] and (res.shape[1] // 3) <= m <= (res.shape[1] * 2 // 3):
                res[n, m] = -0.02
            if (res.shape[0] // 3) <= n <= res.shape[0] and (res.shape[1] * 2 // 3) <= m <= res.shape[1]:
                res[n, m] = 0.05
    return res

"""
def get_motor_right_matrix(shape: Tuple[int, int]) -> np.ndarray:
    # TODO: write your function instead of this one
    res = np.zeros(shape=shape, dtype="float32")
    # these are random values
    res[0:160, 0:] = .45       # Top
    res[160:, 0:213] = .2      # Left
    res[160:320, 213:426] = .2 # Middle
    res[160:, 426:] = .25       # Right

    return res
"""

def get_motor_right_matrix(shape: Tuple[int, int]) -> np.ndarray:
    # Initialize matrix
    res = np.zeros(shape=shape, dtype="float32")
    # Divide image into 9 sections and tune actions accordingly
    for n in range(0, res.shape[0]):
        for m in range(0, res.shape[1]):
            # Top row (object farthest)
            if 0 <= n <= (res.shape[0] // 3) and 0 <= m <= (res.shape[1] // 3):
                res[n, m] = 0.40
            if 0 <= n <= (res.shape[0] // 3) and (res.shape[1] // 3) <= m <= (res.shape[1] * 2 // 3):
                res[n, m] = 0.45
            if 0 <= n <= (res.shape[0] // 3) and (res.shape[1] * 2 // 3) <= m <= res.shape[1]:
                res[n, m] = 0.45
            # Middle row
            if (res.shape[0] // 3) <= n <= (res.shape[0] * 2 // 3) and 0 <= m <= (res.shape[1] // 3):
                res[n, m] = 0.20
            if (res.shape[0] // 3) <= n <= (res.shape[0] * 2 // 3) and (res.shape[1] // 3) <= m <= (res.shape[1] * 2 // 3):
                res[n, m] = 0.25
            if (res.shape[0] // 3) <= n <= (res.shape[0] * 2 // 3) and (res.shape[1] * 2 // 3) <= m <= res.shape[1]:
                res[n, m] = 0.25
            # Bottom row (object closest)
            if (res.shape[0] // 3) <= n <= res.shape[0] and 0 <= m <= (res.shape[1] // 3):
                res[n, m] = 0.05
            if (res.shape[0] // 3) <= n <= res.shape[0] and (res.shape[1] // 3) <= m <= (res.shape[1] * 2 // 3):
                res[n, m] = 0.10
            if (res.shape[0] // 3) <= n <= res.shape[0] and (res.shape[1] * 2 // 3) <= m <= res.shape[1]:
                res[n, m] = 0.10
    return res
"""