import numpy as np
import matplotlib.pyplot as plt


# 找到牛顿插值法的系数
def func_coef(xs, ys):
    n = len(xs)
    if n > 1:
        f = (func_coef(xs[0:n - 1], ys[0:n - 1]) - func_coef(xs[1:n], ys[1:n])) / (xs[0] - xs[n - 1])  # 递归
    else:
        f = ys[0]
    return f


# 得到牛顿插值函数
def newfunc(f, xs, x):
    y = np.array([0 for i in range(len(x))])
    for i in range(len(f)):
        coef = f[i]
        for j in range(i):
            coef = coef * (x - xs[j])
        y = y + coef
    return y


# cubic splines
def cubic(xs, ys):
    n = len(xs)
    left = np.zeros((n, n))
    right = np.zeros((n, 1))
    left[0, 0] = left[n - 1, n - 1] = 1
    # find the solution of f''(x)
    for i in range(1, n - 1):
        left[i, i - 1] = xs[i] - xs[i - 1]
        left[i, i] = 2 * (xs[i + 1] - xs[i - 1])
        left[i, i + 1] = xs[i + 1] - xs[i]
        right[i] = 6 / (xs[i + 1] - xs[i]) * (ys[i + 1] - ys[i]) + 6 / (xs[i] - xs[i - 1]) * (ys[i - 1] - ys[i])
    sol = np.linalg.solve(left, right)

    # interpret
    x = np.array([])
    y = np.array([])
    for i in range(1, n):
        xi = np.linspace(xs[i - 1], xs[i], 10)
        yi = sol[i - 1] * (xs[i] - xi) ** 3 / (6 * (xs[i] - xs[i - 1])) + \
             sol[i] * (xi - xs[i - 1]) ** 3 / (6 * (xs[i] - xs[i - 1])) + \
             (ys[i - 1] / (xs[i] - xs[i - 1]) - sol[i - 1] * (xs[i] - xs[i - 1]) / 6) \
             * (xs[i] - xi) + (ys[i] / (xs[i] - xs[i - 1]) - sol[i] * (xs[i] - xs[i - 1]) / 6) * (xi - xs[i - 1])
        x = np.hstack((x, xi))
        y = np.hstack((y, yi))
    return x, y


def main():
    # cos(x)
    x1 = np.linspace(0, np.pi, 100)
    xs = np.linspace(0, np.pi, 10)
    ys = np.cos(xs)

    # 1/(1+25x^2)
    # x1 =np.linspace(-1,1,100)
    # xs=np.linspace(-1,1,10)
    # ys = 1/(1+25*xs**2)

    f = [func_coef(xs[0:i + 1], ys[0:i + 1]) for i in range(len(xs))]
    y1 = newfunc(f, xs, x1)
    x2, y2 = cubic(xs, ys)

    plt.subplot(1, 2, 1)
    plt.scatter(xs, ys, color='red')
    plt.plot(x1, y1)
    plt.subplot(1, 2, 2)
    plt.scatter(xs, ys, color='red')
    plt.plot(x2, y2)
    plt.show()


if __name__ == '__main__':
    main()
