import numpy as np
from scipy.integrate import quad
x = 0.5
d = 5


repeat_time = 1000000
allowed = 0

for i in range(repeat_time):
    X = np.random.normal(size=[d, d])
    d_ = 1 / np.sqrt(np.sum(np.linalg.inv(X @ X.T)))
    # z_prime = z * normal_list / np.linalg.norm(normal_list)
    if d_ > x:
        allowed += 1

print(allowed / repeat_time)

# theoretical result
# alpha = np.arccos((2 * y**2 - z**2 - x**2)/(z**2 - x**2))
# beta = np.arctan(z / x / np.tan(alpha / 2))
G = lambda y: np.arccos(x/y) * d * y * np.exp(-d / 2 * y**2)
value = quad(G, x, np.inf)[0] * 2 / np.pi
print(value)
