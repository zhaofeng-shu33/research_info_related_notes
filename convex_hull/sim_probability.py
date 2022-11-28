import argparse
import numpy as np
from numpy import random
import pickle
from scipy.optimize import fsolve
from scipy.spatial import ConvexHull
from scipy.integrate import dblquad
from scipy.special import gamma, gammainc
# from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
def interpolation_error(n, d, distribution='gaussian'):
    if distribution == 'gaussian':
        points = np.random.normal(size=(n + 1, d))
    elif distribution == 'uniform':
        points = np.random.normal(size=(n + 1, d))
        norm_list = np.linalg.norm(points, axis=1)
        points = (points.T / norm_list).T
    else:
        raise ValueError("")
    hull = ConvexHull(points)
    return len(hull.vertices) / (n + 1)

def testN(n, d, distribution='gaussian', nTrial=20):
    val = 0
    for i in range(nTrial):
        val += interpolation_error(n, d, distribution)
    return val / nTrial

def _bisection(N_left, N_right, eps, d, distribution, nTrial):
    if (N_right - N_left <= 1):
        return N_left
    N_middle = int((N_left + N_right) / 2)
    p = testN(N_middle, d, distribution, nTrial)
    if p < eps:
        return _bisection(N_left, N_middle, eps, d, distribution, nTrial)
    else:
        return _bisection(N_middle, N_right, eps, d, distribution, nTrial)

def bisection(eps, d, distribution='gaussian', nTrial=50):
    N_left = d + 1
    N_right = 2 ** (d + 1)
    while testN(N_right, d, distribution, nTrial) > eps:
        N_right *= 2
    return _bisection(N_left, N_right, eps, d, distribution, nTrial)
    
    
if __name__ == '__main__':
    d_list = [2, 3, 4, 5, 6, 7]
    eps = 0.5
    N_estimation = []
    for d in d_list:
        print(d)
        N_estimation.append(bisection(eps, d, distribution='uniform'))
    print(N_estimation)
    # Gaussian: [10, 26, 62, 132, 273, 562]

