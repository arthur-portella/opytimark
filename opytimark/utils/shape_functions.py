"""Shape Functions.
"""

import numpy as np

import opytimark.utils.exception as e

def linear(x: np.array, m: int):
    M = len(x) + 1

    result = 1.0

    if np.any((x < 0) | (x > 1)):
        raise e.ValueError("`x` should be in [0,1]")

    if m == 1:
        result = np.prod(x[0 : M-1])
    elif 2 <= m <= M-1:
        result = (np.prod(x[0 : M-m]) * (1 - x[M-m]))
    else:
        result = 1 - x[0]
        
    return result

def convex(x: np.array, m: int):
    M = len(x) + 1
    
    if np.any((x < 0) | (x > 1)):
        raise e.ValueError("`x` should be in [0,1]")

    if m == 1:
        return np.prod(1 - np.cos(x[:M - 1] * np.pi / 2))
    elif m == M:
        return 1 - np.sin(x[0] * np.pi / 2)
    else: 
        return np.prod(1 - np.cos(x[:M - m] * np.pi / 2)) * (1 - np.sin(x[M - m] * np.pi / 2)) 
    

def concave(x: np.array, m: int):
    M = len(x) + 1
    
    if np.any((x < 0) | (x > 1)):
        raise e.ValueError("`x` should be in [0,1]")

    if m == 1:
        return np.prod(np.sin(x[0 : M - 1] * np.pi / 2))
    elif m == M:
        return np.cos(x[0] * np.pi / 2)
    else:
        return (np.prod(np.sin(x[0 : M - m] * np.pi / 2))) * (np.cos(x[M - m] * np.pi / 2))
    
def mixed(x: np.array, alpha: float, A: int):
    
    if np.any((x < 0) | (x > 1)):
        raise e.ValueError("`x` should be in [0,1]")
    
    if alpha <= 0:
        raise e.ValueError("`alpha` should be > 0")
    
    if not isinstance(A, int):
        raise e.TypeError("`A` should be an integer")

    if A < 1:
        raise e.ValueError("`A` should be >= 1")
    
    return (1 - x[0] - ((np.cos(2 * A * np.pi * x[0] + (np.pi / 2))) / (2 * A * np.pi))) ** alpha


def disc(x: np.array, alpha: float, beta: float, A: int):

    if np.any((x < 0) | (x > 1)):
        raise e.ValueError("`x` should be in [0,1]")
     
    if not (alpha > 0 and beta > 0): 
        raise e.ValueError("`alpha` and `beta` should be > 0")

    if A < 1:
        raise e.ValueError("`A` should be >= 1")
        
    if not (isinstance(A, int)):
        raise e.TypeError("`A` should be an integer")
        
    return 1 - (x[0] ** alpha) * (np.cos(A * (x[0] ** beta) * np.pi) ** 2)
