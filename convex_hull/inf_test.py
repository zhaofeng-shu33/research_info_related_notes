import numpy as np
from scipy.integrate import dblquad, quad
x_list = [10,100,500,1000,10000.0]
int_list = []
for u in x_list:
    result = dblquad(lambda x,y: x**4 * np.exp(-y**4/2 - x**2*y**2), u, np.inf, -np.inf, np.inf)
    int_list.append(result[0])
print(int_list)


