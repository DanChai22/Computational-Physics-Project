import numpy as np
import matplotlib.pyplot as plt


def func(theta, phi0):
    f = 1 / np.sqrt(1 - (np.sin(phi0 / 2) * np.sin(theta)) ** 2)
    return f


def itg(phi0, n=100):
    h = np.pi / 2 / n  # step length of phi0s

    x = np.linspace(0, np.pi / 2, n)
    x = np.hstack((x, np.array([np.pi / 2])))

    I = 0
    for i in range(int(n / 2)):
        # Simpson's rule
        I += h / 3 * (func(x[2 * i], phi0) + 4 * func(x[2 * i + 1], phi0) + func(x[2 * i + 2], phi0))

    return I


def main():
    phi0s = np.linspace(-np.pi / 2, np.pi / 2 + np.pi / 100, 100)
    I = np.array([])

    for phi0 in phi0s:
        # calculate the integral with different phi0s
        I = np.append(I, itg(phi0))

    plt.plot(phi0s, I)
    plt.xlabel('phi0')
    plt.ylabel('T(phi0)')
    plt.show()


if __name__ == '__main__':
    main()
