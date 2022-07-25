import numpy as np
from scipy.integrate import quad
from scipy.special import gammainc
x = 0.5
d = 4


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

# theoretical result
# alpha = np.arccos((2 * y**2 - z**2 - x**2)/(z**2 - x**2))
# beta = np.arctan(z / x / np.tan(alpha / 2))
# G = lambda y: np.arccos(x/y) * d * y * np.exp(-d / 2 * y**2)
# value = quad(G, x, np.inf)[0] * 2 / np.pi
k = (d-1)/2
K_x = gammainc(k, x*x/2)
print((1-K_x) * np.exp(-x*x/2))
