import numpy as np
import matplotlib.pyplot as plt


def func(x):
    return np.sin(x)


def phi(h, x):
    return 1 / (2 * h) * (func(x + h) - func(x - h))


def D(n, m, h, x):
    if m == 0:
        d = phi(h / (2 ** n), x)
    else:
        d = D(n, m - 1, h, x) + 1 / (4 ** m - 1) * (D(n, m - 1, h, x) - D(n - 1, m - 1, h, x))
    return d


def main():
    h = 1
    x = np.pi / 3
    itg = 6
    err = 0.0000001
    DM = np.zeros((itg, itg))
    for i in range(itg):
        DM[i,0]=D(i,0,h,x)
    for i in range(itg):
        for j in range(i+1):
            DM[i, j] = D(i, j, h, x)
        if (abs(DM[i, i] - DM[i-1, i-1]) < err) and (i > 1):
            sol = DM[i, i]
            itg = i
            print(sol, itg)
            print(DM[:i,:i])
            break

if __name__ == '__main__':
    main()
