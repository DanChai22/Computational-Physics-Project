import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton

def potential_func(U, E, l=0, option='V1'):
    # U=[U1,U2,r]
    U1, U2, r = U
    if option == 'V1':
        Vr = -1 / r + l * (l + 1) / r ** 2
    elif option == 'V2':
        Vr = -1 / r ** 4 + l * (l + 1) / r ** 2
    else:
        print("Wrong,option should be V1 or V2")
        exit()
    return np.array([U2, 2 * (E - Vr) * U1, 1])


def rk4(U, h, func, *args):
    k1 = func(U, *args)
    k2 = func(U + 0.5 * h * k1, *args)
    k3 = func(U + 0.5 * h * k2, *args)
    k4 = func(U + h * k3, *args)
    return U + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4) * h


def RK4(r_list, U_initial, func, *args):
    h = r_list[1] - r_list[0]
    u = np.zeros((len(r_list), len(U_initial)))
    u[0] = U_initial
    for i in range(len(r_list) - 1):
        u[i + 1] = rk4(u[i], h, func, *args)
    return u


def secant(r_list, U_initial1, U_initial2, func, b, err, *args):
    x = np.vstack((U_initial1, U_initial2))
    f = np.vstack((RK4(r_list, x[-2], func, *args)[-1], RK4(r_list, x[-1], func, *args)[-1]))
    while 1:
        xtemp = x[-1, 1] - f[-1, 0] * (x[-1, 1] - x[-2, 1]) / (f[-1, 0] - f[-2, 0])
        xtemp2 = [U_initial1[0], xtemp, U_initial1[2]]
        x = np.vstack((x, xtemp2))
        f = np.vstack((f, RK4(r_list, x[-1], func, *args)[-1]))
        if (f[-1, 0] - b) < err:
            break
    return x[-1]


def shooting(a, b, ui1, ui2, func, r_list, err, *args):
    U_initial1 = [a, ui1, r_list[0]]
    U_initial2 = [a, ui2, r_list[0]]
    Uftemp = secant(r_list, U_initial1, U_initial2, func, b, err, *args)
    Uf = RK4(r_list, Uftemp, func, *args)
    return Uf


if __name__ == "__main__":
    r_list = np.linspace(100, 1e-10, 2000)
    E_list = np.arange(-1, -1e-4, 100)
    l = 0
    u = shooting(0, 0, 0.1, 0.2, potential_func, r_list, 1e-5, -1, l, 'V1')
    print(u)
