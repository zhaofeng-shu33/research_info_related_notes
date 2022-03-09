from scipy.spatial import ConvexHull  
import numpy as np
import matplotlib.pyplot as plt
d = 4
N = 45
num_points = 1000
f_list = np.zeros(num_points)
v_list = np.zeros(num_points)

for i in range(num_points):
    p = np.random.random([N, d])
    hull=ConvexHull(p)
    f=hull.nsimplex
    v=len(hull.vertices)
    f_list[i] = f
    v_list[i] = v
plt.scatter(v_list, f_list)
plt.show()
