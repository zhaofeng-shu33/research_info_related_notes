import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(0.01, 10)
def f(x, n):
    return n * np.power(1-np.exp(-x), n-1) * np.exp(-x)
y1 = f(x, 1)
y2 = f(x, 2)
y3 = f(x, 3) 
y10 = f(x, 10)
plt.plot(x, y1, label='n=1')
plt.plot(x, y2, label='n=2')
plt.plot(x, y3, label='n=3')
plt.plot(x, y10, label='n=10')
plt.legend()
plt.show()
