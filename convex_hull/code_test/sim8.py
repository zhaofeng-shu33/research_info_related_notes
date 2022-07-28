import numpy as np
from scipy.special import gamma, beta
q = 2
k = q + 1
d = 3
L = np.power(2, q+1) / (q+1) / beta(d/2, q+1)

a = np.power(2, (d-1)/2) * k * gamma(d/2) / (d-1) / np.sqrt(np.pi) / gamma((d-1)/2) * beta(k, (d+1)/2)
b = k**d / np.pi * np.power(2, 0.5+d*(d/2-1)) * np.power(beta(k, d/2), d) * beta(0.5, d*(k+d/2 - 1) + 1)
F = b / gamma(d+1) * np.power(a, -d+(d-1)/(2*k+d-1)) * gamma(d+1-(d-1)/(2*k+d-1)) * np.power(L, (d-1)/(2*k+d-1))

c3 = 2 * gamma((d*d+2*d*q+1)/(d+2*q+1)) / (d+2*q+1) / gamma(d+1) / np.sqrt(np.pi)
c3 *= gamma((d*d+2*d*q+2)/2) / gamma((d*d+2*d*q+1)/2)
c3 *= np.power(2*np.pi/beta(0.5, d/2+q+1), (d*d+2*d*q+1)/(d+2*q+1))
print(c3)
print(F)
