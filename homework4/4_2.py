import numpy as np
import matplotlib.pyplot as plt

p = np.array([98500, 110800, 136300, 163100])
V = np.array([0.025, 0.0222, 0.018, 0.015])
X = np.vstack((1 / V, 1 / V ** 2)).T
Y = p * V / 8.314 / 303 - 1
b = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y)#找到系数a，b
print(b)

def result_func(x, A=b):
    return (A[0] / x + A[1] / x ** 2 + 1) * 8.314 * 303 / x


x = np.linspace(0.014, 0.026, 100)
y = result_func(x)

plt.plot(x, y, color="green")
plt.xlabel('V/m^3',fontsize=15)
plt.ylabel('p/Pa',fontsize=15)
plt.scatter(V, p)
plt.show()
