import numpy as np
import matplotlib.pyplot as plt
if __name__ == '__main__':
    d_list = np.array([2, 3, 4, 5, 6, 7])
    uniform = [3, 6, 20, 48, 123, 323]
    estimated_N_uniform = np.power(2 * np.pi, d_list / 2) * np.power(d_list, d_list / 2 - 2.5) / 0.9
    estimated_N_uniform = np.power(d_list, (d_list - 1) / 2) * np.power(estimated_N_uniform, (d_list + 1)/2)
    plt.xlabel('d')
    plt.ylabel('log(N)') # empirical N to reach p = 0.5
    plt.plot(d_list, np.log(uniform), label='empirical', marker='+')
    plt.plot(d_list, np.log(estimated_N_uniform), label='bound', marker='x')
    plt.legend()
    plt.title('uniform')
    plt.savefig('uniform.pdf')
    plt.show()
