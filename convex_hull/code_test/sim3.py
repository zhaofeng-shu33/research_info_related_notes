import numpy as np
x = 0.6
y = x*np.sqrt(2)
z = 1
# x = 1
# y = 2
# z = 3
phi_2 = np.arcsin(x/y)
k = np.tan(phi_2)
def check_in(point):
    bool_1 = point[1] + k * (point[0] - y) > 0
    bool_2 = point[1] - k * (point[0] - y) < 0
    return bool_1 and bool_2

repeat_time = 100000
allowed = 0

for i in range(repeat_time):
    normal_list = np.random.normal(size=3)
    z_prime = z * normal_list / np.linalg.norm(normal_list)
    if check_in(z_prime):
        allowed += 1

print(allowed / repeat_time)

# theoretical result
alpha = np.arccos((2 * y**2 - z**2 - x**2)/(z**2 - x**2))
beta = np.arctan(z / x / np.tan(alpha / 2))

ratio = (np.pi - 2 * beta -  alpha * x / z) / (4 * np.pi)
print(ratio * 2)
