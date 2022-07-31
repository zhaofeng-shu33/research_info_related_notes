import numpy as np
from scipy.integrate import quad, dblquad, tplquad
from scipy.special import gammainc,gamma
x = 0.5
d = 5



def f_g(r): # pdf of radius 3d gaussian
    return  r**(d-1) * np.exp(-r**2/2) * 2**(1-d/2) / gamma(d/2)

# theoretical result
UPPER = 30
G = lambda z1, z, y: 2*x**2/y**3 * (2* np.sqrt(1-y*y/z**2)*np.sqrt(1-y*y/z1**2)-((y*y/z/z1+np.sqrt(1-y*y/z1**2)*np.sqrt(1-y*y/z**2))**3-(y*y/z/z1-np.sqrt(1-y*y/z1**2)*np.sqrt(1-y*y/z**2))**3)/3) * f_g(z) * f_g(z1)
val = tplquad(G, x, UPPER, lambda y: y, lambda y: UPPER, lambda y,z: y, lambda y,z: z)
print(val[0] * 2 * gamma(d/2) / np.sqrt(np.pi) / gamma((d-1)/2))
print(np.exp(-x*x))
