import numpy as np
import math


def function(X):
    if np.dot(X, X) < 1:
        f = 1
    else:
        f = 0
    return f


def gauss(n, X, sigma, mu):
    gauss = 1 / (math.sqrt(2 * math.pi) * sigma) ** n * math.exp(-0.5 * np.dot(X - mu, X - mu) / sigma ** 2)
    return gauss


def findSigma(sigma=0.05, n=20):
    num_samples = 1000
    a = 1
    while a:
        num1 = 0
        num0 = 0
        for i in range(num_samples):
            X = np.random.normal(0, sigma, size=n)
            num1 += function(X)
            num0 += (1 - function(X))
        if (abs(num0 - num1) > 90) & (sigma < 1):
            sigma += 0.001
        else:
            a = 0
    return sigma


def Integral(num_samples: int = 100000, n: int = 20) -> float:
    sum = 0
    sigma = findSigma(sigma=0.05, n=n) + 0.005
    for i in range(num_samples):
        X = np.random.normal(0, sigma, size=n)
        fi = function(X) / gauss(n, X, sigma, 0)
        sum += fi
    sum = sum / num_samples
    return sum


if __name__ == '__main__':
    sum = Integral(num_samples=100000, n=5)
    # print(type(sample))
    print('%.15f' % sum)
