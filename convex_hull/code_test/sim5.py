import numpy as np
from scipy.integrate import quad
x = 0.5
d = 4


repeat_time = 100000
allowed = 0

for i in range(repeat_time):
    X = np.random.normal(size=[3, 4])
    a_ = X[0, :]
    b_ = X[1, :]
    c_ = X[2, :]
    a = a_ - b_
    b = c_ - b_
    cos_theta = np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)
    sin_theta = np.sqrt(1 - cos_theta ** 2)
    triangle_area = 0.5 * np.linalg.norm(a) * np.linalg.norm(b) * sin_theta
    tmp_matrix = np.zeros([4, 4])
    tmp_vector = np.zeros(4)
    tmp_matrix[1:, :] = X
    for i in range(4):
        tmp_matrix[0, :] = 0
        tmp_matrix[0, i] = 1
        tmp_vector[i] = np.linalg.det(tmp_matrix)
    tetrahedron_volume = np.linalg.norm(tmp_vector) / 6
    h = 3 * tetrahedron_volume / triangle_area
    # z_prime = z * normal_list / np.linalg.norm(normal_list)
    if h > x:
        allowed += 1

print(allowed / repeat_time)

# theoretical result
# alpha = np.arccos((2 * y**2 - z**2 - x**2)/(z**2 - x**2))
# beta = np.arctan(z / x / np.tan(alpha / 2))
# G = lambda y: np.arccos(x/y) * d * y * np.exp(-d / 2 * y**2)
# value = quad(G, x, np.inf)[0] * 2 / np.pi
K_x = np.exp(-x*x/2)
print(K_x**3)
