from operator import mod
import numpy as np
from scipy.special import comb
from scipy.special import beta
from scipy.integrate import quad
a = 1.3
b = 2.2
c = np.sqrt(a/b)
d = 6
n = d - 2
def g(k):
    f1 = lambda x: np.power(b-x, (k-1)/2) * np.power(a-x, (d-2-k)/2) * np.power(x, (d-3)/2)
    return quad(f1, 0, a)[0]
# left = 0
# for k in range(0, d-2+1):
#    left += comb(d-2, k) * g(k+mod(d-k+1, 2))
# left = 0
# for t in range(0, int((d-2)/2) + 1):
#    left += comb(d-1, 2*t) * quad(lambda x: np.power(b-x, t) * np.power(a-x, (d-1-2*t)/2) * np.power(x, (d-5)/2), 0, a)[0]
# left *= (d-3)/(d-1)
f = lambda t: ((1+t)**(d-1) - (1-t)**(d-1)) * 2 * t * (a-b*t*t)**((d-5)/2) / (1-t*t)**(d-1)
left = 0.5* (d-3)/(d-1) * (b-a)**((d+1)/2) * quad(f, 0, np.sqrt(a/b))[0]
right = beta((d-1)/2, 0.5) * (a*b)**((d-2)/2)
# print(left, right)
left = quad(lambda t: (1/((1-t)**(n+2)) + 1/((1+t)**(n+2)))*(c*c-t*t)**((n-1)/2), 0, c)[0]
right = beta((d-1)/2, 0.5) * c**n / (1-c*c)**((n+3)/2)
print(left, right)
