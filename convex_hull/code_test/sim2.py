import numpy as np
from scipy import integrate

x = 1
UPPER = np.inf
def f(r):
    return 4 * r**2 / (1+r**2)**2 / np.pi

G = lambda z1, z, y: f(z1) * f(z) * f(y) * (1-x/z1) * np.sqrt(1-x**2/z**2) * np.sqrt(1-x**2/y**2) 
val = integrate.tplquad(G, x, UPPER, lambda z1: z1, lambda z1: UPPER,
lambda z1, z: z, lambda z1, z: UPPER)
H_x = val[0] * 3
print(H_x)


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

repeat_time = 10000

# generate random points on sphere with radius z
allowed = 0
for i in range(repeat_time):
    points = random_points_3d_cauchy(3)
    a = points[0, :]
    b = points[1, :]
    c = points[2, :]
    square = 2 * np.linalg.norm(np.cross(a - c, b - c))
    z_prime = z * normal_list / np.linalg.norm(normal_list)
    # test whether the distance from the origin to the line z_prime, y is larger
    # than x
    h = np.linalg.norm(np.cross(z_prime, y_prime)) / np.linalg.norm(z_prime - y_prime)
    allowed += (h > x)
print(allowed / repeat_time)