import numpy as np
import matplotlib.pyplot as plt

'''
本程序是利用hybrid method来求解方程 tanh(x/t)-x=0的根,给出x(t)
定义了三个函数func(x,t),dfunc(x,t),hybrid(a, b, t, err=1E-14)
func(x,t),dfunc(x,t)分别给出了tanh(x / t) - x及其导数-(tanh(x / t) )^2
hybrid(a, b, t, err=1E-14)给出了在初始区间[a,b]下使用hybrid method,误差为err的根。
'''


def func(x, t):
    return np.tanh(x / t) - x


def dfunc(x, t):
    return -np.tanh(x / t) ** 2


def hybrid(a, b, t, err=1E-14):
    x0 = 0.5 * (a + b)
    if func(x0, t) == 0:
        x0 = x0
    while 1:
        x0 = x0 - func(x0, t) / dfunc(x0, t)
        if x0 < a or x0 > b:
            x0 = (a + b) * 0.5
            if func(a, t) * func(x0, t) > 0:
                a = x0
            if func(b, t) * func(x0, t) > 0:
                b = x0

        if abs(func(x0, t)) < err:
            break
    return x0


if __name__ == '__main__':
    x0 = []
    for i in range(1, 200):
        t = i * 0.01
        x0.append(hybrid(-1.1, 1.2, t))

    t=np.zeros(199)
    for i in range(1,200):
        t[i-1] = i*0.01

    f = plt.figure(figsize=(9, 5))
    plt.plot(t, x0, 'o-', color="green")
    plt.xlabel("t", fontsize=20)
    plt.ylabel("m(t)", fontsize=20)
    plt.show()