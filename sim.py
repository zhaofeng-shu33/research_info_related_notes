import numpy as np
p_0 = 0.5
p_1 = 0.4

def f(a):
    C_11 = np.power(p_1, 1 - a) * np.power(p_0, a)
    C_12 = np.power(1 - p_1, 1 - a) * np.power(1 - p_0, a)
    C_21 = np.power(p_0, 1 - a) * np.power(p_1, a)
    C_22 = np.power(1 - p_0, 1 - a) * np.power(1 - p_1, a)
    res_1 = C_11 * np.log(p_0 / p_1)
    res_1 += C_12 * np.log((1 - p_0) / (1 - p_1))
    res_1 /= (C_11 + C_12)
    res_2 = C_21 * np.log(p_1 / p_0)
    res_2 += C_22 * np.log((1 - p_1) / (1 - p_0))
    res_2 /= (C_21 + C_22)
    res_3 = -1 * np.log(C_11 + C_12) - np.log(C_21 + C_22)
    return a * res_1 + a * res_2 +  res_3

def g(a):
    res = 2 * a * np.log(a / p_0)
    res += 2 * (1 - a) * np.log((1 - a) / (1 - p_0))
    res -= a * np.log(a / p_1)
    res -= (1 - a) * np.log((1 - a) / (1 - p_1))
    return res
