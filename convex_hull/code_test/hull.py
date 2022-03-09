from scipy.spatial import ConvexHull  
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
d = 4
N = 45
num_points = 1000
f_list = np.zeros(num_points)
v_list = np.zeros(num_points)
f2_list = np.zeros(num_points) # edge list
f3_list = np.zeros(num_points)

def get_low_dim_facet(simplex_list, d=2):
    E_list = set()
    for simplex in simplex_list:
        for facet in combinations(simplex, d):
            E_list.add(tuple(sorted(facet)))
    return E_list

for i in range(num_points):
    p = np.random.random([N, d])
    hull=ConvexHull(p)
    f=hull.nsimplex
    f2_list[i] = len(get_low_dim_facet(hull.simplices, 2))
    f3_list[i] = len(get_low_dim_facet(hull.simplices, 3))
    v=len(hull.vertices)
    f_list[i] = f
    v_list[i] = v
# print(v_list - f2_list + f3_list - f_list)
plt.scatter(f2_list, v_list, label='f2-f1')
plt.scatter(f3_list, f2_list, label='f3-f2')
plt.scatter(f_list, f3_list, label='f4-f3')
plt.legend()
plt.show()
