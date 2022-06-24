import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(0.01, 10)
planck = 1 / (np.exp(1/x) - 1)
schrodinger = planck + 0.5
classical = x
plt.plot(x, planck, label='planck')
plt.plot(x, schrodinger, label='schrödinger')
plt.plot(x, classical, label='classical')
plt.legend()
plt.xlabel('$\\frac{kT}{\\hbar \\omega}$', size=16)
plt.ylabel('$\\frac{U}{\\hbar \\omega}$',rotation=0, size=16)
plt.title("The mean energy U of a simple harmonic oscillator \n as a function of temperature")
plt.show()
