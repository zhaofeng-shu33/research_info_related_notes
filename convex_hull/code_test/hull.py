from scipy.spatial import ConvexHull  
from sklearn.linear_model import LinearRegression
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
d = 4
num_points = 100
N = 50
v_list = np.zeros(num_points) # vertices list
f2_list = np.zeros(num_points) # edge list
f_list = np.zeros(num_points) # volume list
f3_list = np.zeros(num_points) # face list

def get_low_dim_facet(simplex_list, d=2):
    E_list = set()
    for simplex in simplex_list:
        for facet in combinations(simplex, d):
            E_list.add(tuple(sorted(facet)))
    return E_list

def test_linear(x, y):
    reg = LinearRegression().fit(x, y)
    print(reg.coef_)
    print(reg.intercept_)
    return reg.score(x, y)

for i in range(num_points):
    p = np.random.random([N, d])
    hull = ConvexHull(p)
    f = hull.nsimplex
    f2_list[i] = len(get_low_dim_facet(hull.simplices, 2))
    f3_list[i] = len(get_low_dim_facet(hull.simplices, 3))
    v = len(hull.vertices)
    f_list[i] = f
    v_list[i] = v

# print(v_list - f2_list + f3_list - f_list)
_score = test_linear(np.array(v_list).reshape(-1, 1), f2_list)
print(_score)
# plt.scatter(v_list_total, f2_list_total, label='f2-f1')
# plt.scatter(f3_list, f2_list, label='f3-f2')
plt.scatter(np.arange(num_points), f_list / v_list)
# plt.scatter(v_list, f_list, label='f4-f3')
plt.legend()
plt.show()
