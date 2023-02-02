from scipy.integrate import dblquad, quad
from scipy.stats import norm
import numpy as np
r = -0.1
n = 3
f = lambda y, x: 1/(2*np.pi*np.sqrt(1+n*r) *np.power(1+x*x+y*y-r/(1+2*r)*(x+y)**2,1.5))
g_2_r = dblquad(f, 0, np.inf, lambda x: 0, lambda x: np.inf)[0]
# f_1 = lambda x: (0.5+np.arctan(np.sqrt(r)*x)/np.pi)**2 * np.power(r*x*x+1, -(1/r+1)/2)
# print(quad(f_1,-np.inf, np.inf)[0])
f_star = lambda x: (np.power(complex(0.5, quad(lambda y: np.exp(y**2),0,x*np.sqrt(-r))[0]/np.sqrt(np.pi)),n)).real * np.exp(-x**2)
g_r = quad(f_star, 0, 30)[0] * 2 / np.sqrt(np.pi)
print(g_r)
print(g_2_r)
