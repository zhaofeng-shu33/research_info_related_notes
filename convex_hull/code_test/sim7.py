import numpy as np

y = 3
a = np.array([0, 0, 0, y])
x = 1
z = 2
repeat_time = 1000000
# generate random points on sphere with radius z
allowed = 0
for i in range(repeat_time):
    normal_list = np.random.normal(size=4)
    b = z * normal_list / np.linalg.norm(normal_list)
    # test whether the distance from the origin to the line z_prime, y is larger
    # than x
    cos_theta = np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)
    sin_theta = np.sqrt(1 - cos_theta ** 2)
    h = np.linalg.norm(a) * np.linalg.norm(b) * sin_theta / np.linalg.norm(a-b)
    allowed += (h > x)
print(allowed / repeat_time)

print(2/np.pi *(np.arccos(x/z) + np.sqrt(1-x*x/(z*z)) * x/z * (1-2*x*x/(y*y))))