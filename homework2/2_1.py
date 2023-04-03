import numpy as np
import matplotlib.pyplot as plt

'''
本程序是利用bisection method,Newton-Raphson method, hybrid method来求解方程 x^3 − 5x + 3 = 0的根
定义了五个函数func(x),dfunc(x),bisection(a,b,err=1E-4),newton(x0, err=1E-14),hybrid(a, b, err=1E-14)
func(x),dfunc(x)分别给出了x^3 − 5x + 3及其导数3x^2-5
bisection(a,b,err=1E-4)给出了在初始区间[a,b]下使用bisection method,误差为err的根。
newton(x0, err=1E-14)给出了初值取为x0，使用Newton-Raphson method,误差为err的根。
hybrid(a, b, err=1E-14)给出了在初始区间[a,b]下使用hybrid method,误差为err的根。
'''

def func(x):
    return x ** 3 - 5 * x + 3


def dfunc(x):
    return 3 * x ** 2 - 5


def bisection(a, b, err=1E-4):
    while 1:
        x0 = 0.5 * (a + b)
        if func(x0) == 0:
            break
        if func(a) * func(x0) > 0:
            a = x0
        if func(b) * func(x0) > 0:
            b = x0
        if func(x0 + err) * func(x0 - err) < 0:
            print(x0)
            break


def newton(x0, err=1E-14):
    while 1:
        if func(x0 + err) * func(x0 - err) < 0:
            print(x0)
            break
        else:
            x0 = x0 - func(x0) / dfunc(x0)


def hybrid(a, b, err=1E-14):
    x0 = 0.5 * (a + b)
    if func(x0) == 0:
        print(x0)
    while 1:
        x0 = x0 - func(x0) / dfunc(x0)
        if x0 < a or x0 > b:#如果迭代值落在原区间外，则使用二分法
            x0 = (a + b) * 0.5
            if func(a) * func(x0) > 0:
                a = x0
            if func(b) * func(x0) > 0:
                b = x0

        if func(x0 + err) * func(x0 - err) < 0:
            print(x0)
            break


if __name__ == '__main__':

    print("solution with bisection")
    bisection(0, 1)
    bisection(1, 2)
    print("solution with newton")
    newton(0.66)
    newton(1.84)
    print("solution with hybrid")
    hybrid(0, 1)
    hybrid(1, 2)

    x = np.linspace(-5, 5, 100)
    y = func(x)
    plt.subplot(121)
    plt.plot(x, y)
    plt.axhline(y=0, color='black')
    plt.show()
