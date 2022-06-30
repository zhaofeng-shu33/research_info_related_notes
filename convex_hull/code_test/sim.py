import numpy as np

y = 3
y_prime = np.array([0, 0, y])
x = 1
z = 2
repeat_time = 1000000
# generate random points on sphere with radius z
allowed = 0
for i in range(repeat_time):
    normal_list = np.random.normal(size=3)
    z_prime = z * normal_list / np.linalg.norm(normal_list)
    # test whether the distance from the origin to the line z_prime, y is larger
    # than x
    h = np.linalg.norm(np.cross(z_prime, y_prime)) / np.linalg.norm(z_prime - y_prime)
    allowed += (h > x)
print(allowed / repeat_time)

print(np.sqrt(1-(x/y)**2) * np.sqrt(1-(x/z)**2))