import numpy as np
import matplotlib.pyplot as plt
if __name__ == '__main__':
    d_list = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10])
    cauchy = [7, 14, 21, 30, 40, 51, 60, 75, 87]
    estimated_N_cauchy = np.power(np.pi, d_list) / d_list ** 1.5 / 0.5
    plt.xlabel('d')
    plt.ylabel('log(N)') # empirical N to reach p = 0.5
    plt.plot(d_list, np.log(cauchy), label='empirical', marker='+')
    plt.plot(d_list, np.log(estimated_N_cauchy), label='bound', marker='x')
    plt.legend()
    plt.title('cauchy')
    plt.savefig('cauchy.pdf')
    plt.show()
