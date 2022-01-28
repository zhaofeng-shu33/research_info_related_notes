import numpy as np
from scipy.special import gamma

n = 3
def B(p):
    return 2 * (np.pi) ** (p/2) / (p * gamma(p / 2))

result = 2
result *= B(n**2 - 1) / B(n ** 2)
result *= gamma((n**2+1)/(n+1)) / gamma(n+2)
result *= ((n+1) * B(n) / B(n-1)) ** ((n**2+1)/(n+1))
print(result)
print(35*np.sqrt(3) * gamma(5/2) / 9)