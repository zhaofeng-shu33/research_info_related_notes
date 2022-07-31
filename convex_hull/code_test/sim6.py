import numpy as np
from scipy.integrate import quad, dblquad
from scipy.special import gammainc,gamma
x = 0.5
d = 5


repeat_time = 100000
allowed = 0

for i in range(repeat_time):
    X = np.random.normal(size=[2, d])
    a = X[0, :]
    b = X[1, :]
    cos_theta = np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)
    sin_theta = np.sqrt(1 - cos_theta ** 2)
    h = np.linalg.norm(a) * np.linalg.norm(b) * sin_theta / np.linalg.norm(a-b)
    if h > x:
        allowed += 1

print(allowed / repeat_time)

def f_g(r): # pdf of radius 3d gaussian
    return  r**(d-1) * np.exp(-r**2/2) * 2**(1-d/2) / gamma(d/2)

# theoretical result
UPPER = 30
G = lambda z, y: (2* np.sqrt(1-x*x/y**2)*np.sqrt(1-x*x/z**2)-((x*x/y/z+np.sqrt(1-x*x/z**2)*np.sqrt(1-x*x/y**2))**3-(x*x/y/z-np.sqrt(1-x*x/z**2)*np.sqrt(1-x*x/y**2))**3)/3) * f_g(y) * f_g(z)
val = dblquad(G, x, UPPER, lambda y: x, lambda y: y)
print(val[0] * 2 * gamma(d/2) / np.sqrt(np.pi) / gamma((d-1)/2))
