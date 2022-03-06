import numpy as np
from scipy.integrate import dblquad
from scipy.special import gamma
import matplotlib.pyplot as plt

def g(v):
    return 4 * np.sqrt(np.pi) * gamma(v+ 1/2) / gamma(v+1) * gamma(v/2 + 1) ** 2 / gamma((v+1)/2) ** 2

if __name__ == '__main__':
    v_list = np.linspace(0.01, 5)
    g_v_list = g(v_list)
    plt.plot(v_list, g_v_list)
    plt.scatter([1], [np.pi**2/2], color='red')
    plt.savefig('2d_t_distribution_g_v.pdf')
    plt.show()
