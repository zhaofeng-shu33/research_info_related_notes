from scipy.integrate import dblquad, quad
import numpy as np
r = 0.1
f = lambda y, x: 1/(2*np.pi*np.sqrt(1+2*r) *np.power(1+x*x+y*y-r/(1+2*r)*(x+y)**2,1.5))
g_2_r = dblquad(f, 0, np.inf, lambda x: 0, lambda x: np.inf)[0]
# f_1 = lambda x: (0.5+np.arctan(np.sqrt(r)*x)/np.pi)**2 * np.power(r*x*x+1, -(1/r+1)/2)
# print(quad(f_1,-np.inf, np.inf)[0])
print(np.arccos(-r/(1+r))/(2*np.pi))
print(g_2_r)
