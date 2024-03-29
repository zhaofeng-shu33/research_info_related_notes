import numpy as np
import argparse
import scipy
from scipy import integrate
from scipy.optimize import fsolve



def F_1(r):
    return (r/(2*(r*r+1))+np.arctan(r)/2)*4/np.pi


def random_points_3d_cauchy(n):
    y = np.random.random(n)
    r = np.zeros(n)
    for i in range(n):
        f = lambda r: (-r/(2*(r*r+1))+np.arctan(r)/2)*4/np.pi - y[i]
        r[i] = fsolve(f, [1.0])[0]
    phi = 2 * np.pi * np.random.random(n)
    theta = np.pi * np.random.random(n)
    return np.vstack((r * np.cos(phi) * np.sin(theta),
                      r * np.sin(phi) * np.sin(theta),
                      r * np.cos(theta))).T

def f(r): # pdf of radius 3d cauchy
    return 4 * r**2 / (1+r**2)**2 / np.pi

def f_g(r): # pdf of radius 3d gaussian
    return  r**2 * np.exp(-r**2/2) * np.sqrt(2/np.pi)

def k(x,y,z):
    return 2/np.pi * np.arctan(z/x * np.sqrt(y*y-x*x)/np.sqrt(z*z-y*y)) +\
        x/z * (1/np.pi * np.arccos((2*y*y-z*z-x*x) / (z*z-x*x))-1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--distribution', choices=['gaussian', 'cauchy'], default='gaussian')
    parser.add_argument('--repeat_times', type=int, default=100000)
    args = parser.parse_args()
    x = 1.0
    if args.distribution == 'gaussian':
        print(1 - scipy.special.erf(np.sqrt(3/2) * x))
    else:
        print(1 - F_1(x)) # ground truth value

    if True:
        repeat_time = args.repeat_times

        # generate random points on sphere with radius z
        allowed = 0
        for i in range(repeat_time):
            if args.distribution == 'gaussian':
                points = np.random.normal(size=(3, 3))
            else:
                points = random_points_3d_cauchy(3)
            a = points[0, :]
            b = points[1, :]
            c = points[2, :]
            triangle_square = np.linalg.norm(np.cross(a - c, b - c)) / 2
            tetrahedron_volume = np.abs(np.linalg.det(points)) / 6
            h = 3 * tetrahedron_volume / triangle_square
            
            # test whether the distance from the origin to the plane is larger
            # than x
            allowed += (h > x)
        print(allowed / repeat_time)


    UPPER = 30

    if args.distribution == 'gaussian':
        G = lambda z, y: 2 * y * np.exp(-y*y) * f_g(z) * (1-x/z) 
        val = integrate.dblquad(G, x, UPPER, lambda y: x, lambda y: y)
        G_1 = lambda z, y: 2 * y * np.exp(-y*y) * f_g(z) * k(x,y,z)
        val_1 = integrate.dblquad(G_1, x, UPPER, lambda y: y, lambda y: UPPER)
        H_x = val[0] + val_1[0]
    else:
        G = lambda z1, z, y: f(z1) * f(z) * f(y) * (1-x/z1) * np.sqrt(1-x**2/z**2) * np.sqrt(1-x**2/y**2) 
        val = integrate.tplquad(G, x, UPPER, lambda y: x, lambda y: y,
        lambda y, z: x, lambda y, z: z)
        H_x = val[0] * 6
    print(H_x)