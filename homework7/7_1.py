import numpy as np
import matplotlib.pyplot as plt


def euler(t, x0, v0, omega2):
    h = t[1] - t[0]
    x = np.array([x0])
    v = np.array([v0])

    for i in range(len(t) - 1):
        xtemp = x[i] + h * v[i]
        vtemp = v[i] - omega2 * h * x[i]
        x = np.append(x, xtemp)
        v = np.append(v, vtemp)
    return v, x


def mid(t, x0, v0, omega2):
    h = t[1] - t[0]
    x = np.array([x0])
    v = np.array([v0])

    for i in range(len(t) - 1):
        xmid = v[i] * h / 2 + x[i]
        vmid = v[i] - omega2 * h / 2 * x[i]
        xtemp = x[i] + h * vmid
        vtemp = v[i] - omega2 * h * xmid
        x = np.append(x, xtemp)
        v = np.append(v, vtemp)
    return v, x


def rk4(t, x0, v0, omega2):
    h = t[1] - t[0]
    x = np.array([x0])
    v = np.array([v0])

    for i in range(len(t) - 1):
        xk1 = v[i]
        vk1 = -omega2 * x[i]

        xk2 = v[i] + 0.5 * h * vk1
        vk2 = -omega2 * (x[i] + xk1 * h / 2)

        xk3 = v[i] + 0.5 * h * vk2
        vk3 = -omega2 * (x[i] + xk2 * h / 2)

        xk4 = v[i] + vk3 * h
        vk4 = -omega2 * (x[i] + xk3 * h)

        xtemp = x[i] + h * (xk1 + 2 * xk2 + 2 * xk3 + xk4) / 6
        vtemp = v[i] + h * (vk1 + 2 * vk2 + 2 * vk3 + vk4) / 6

        x = np.append(x, xtemp)
        v = np.append(v, vtemp)
    return v, x


def euler_trapezoidal(t, x0, v0, omega2, error):
    h = t[1] - t[0]
    x = np.array([x0])
    v = np.array([v0])

    for i in range(len(t) - 1):
        x_1 = x[i] + v[i] * h
        v_1 = v[i] - omega2 * x[i] * h

        x_2 = x[i] + h / 2 * (v[i] + v_1)
        v_2 = v[i] - omega2 * (x[i] + x_1) * h / 2

        xc = [x_1, x_2]
        vc = [v_1, v_2]

        while abs(xc[-1] - xc[-2]) > error or abs(vc[-1] - vc[-2]) > error:
            x_j = x[i] + h / 2 * (v[i] + vc[-1])
            v_j = v[i] - omega2 * h / 2 * (x[i] + xc[-1])
            xc.append(x_j)
            vc.append(v_j)

        xtemp = xc[-1]
        vtemp = vc[-1]

        x = np.append(x, xtemp)
        v = np.append(v, vtemp)
    return v, x


def main():
    omega2 = 1
    t = np.linspace(0, 60, 10000)
    # v, x = euler(t, 1, 0, omega2)
    v, x = mid(t, 1, 0, omega2)
    # v, x = rk4(t, 1, 0, omega2)
    # v, x = euler_trapezoidal(t,1,0,omega2,1e-9)

    plt.figure(figsize=(12, 6), dpi=100)
    plt.subplot(1, 2, 1)
    plt.plot(t, x)
    plt.xlabel('t')
    plt.ylabel('angle')
    plt.subplot(1, 2, 2)
    plt.plot(t, 0.5 * omega2 * x ** 2 + 0.5 * v ** 2)
    plt.xlabel('t')
    plt.ylabel('E')
    plt.suptitle('Euler', x=0.5)

    plt.show()


if __name__ == '__main__':
    main()
