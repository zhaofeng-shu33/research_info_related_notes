import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
def f(N, d):
    return N/(np.log(N) **((d-1)/2)) - np.power(2*np.sqrt(np.pi), d) / np.power(d, 1.5)

if __name__ == '__main__':
    d_list = np.array([2, 3, 4, 5, 6, 7])
    gaussian = [10, 26, 62, 132, 273, 562]
    N = np.zeros(len(d_list))
    for i in range(len(d_list)):
        x_0 = np.power(2*np.sqrt(np.pi), d_list[i]) / np.power(d_list[i], 1.5)
        N[i] = fsolve(lambda x: f(x, d_list[i]), [x_0])[0]    
    # estimated_N_cauchy = np.power(np.pi, d_list) / d_list ** 1.5 / 0.5
    plt.xlabel('d')
    plt.ylabel('log(N)') # empirical N to reach p = 0.5
    plt.plot(d_list, np.log(gaussian), label='empirical')
    plt.plot(d_list, np.log(N), label='bound')
    plt.legend()
    plt.title('gaussian')
    plt.savefig('gaussian.pdf')
    plt.show()
