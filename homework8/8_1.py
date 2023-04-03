import numpy as np
import matplotlib.pyplot as plt


def rhof(option):
    if option == 1:
        return 0
    elif option == 2:
        return 1
    else:
        print("Wrong,option should be 1 or 2")
        exit()


def bondcondition(nx, ny, option):
    phi = np.zeros((nx + 1, ny + 1))
    if option == 1:
        for i in range(nx + 1):
            phi[i, -1] = 1
        return phi
    elif option == 2:
        return phi
    else:
        print("Wrong,option should be 1 or 2")
        exit()


def relax_method(iter, nx, ny, h, option):
    phi = bondcondition(nx, ny, option)
    rho = rhof(option)
    for i in range(iter):
        for x in range(1, nx):
            for y in range(1, ny):
                phi[x, y] = 0.25 * (phi[x - 1, y] + phi[x + 1, y] + phi[x, y + 1] + phi[x, y - 1]) \
                            + 0.25 * h ** 2 * rho
    return phi


def main():
    # system parameters
    Lx = 1  # m
    Ly = 1.5  # m
    h = 0.01  # m
    nx = int(Lx / h)
    ny = int(Ly / h)
    option = 1
    iter = 2000

    phi = relax_method(iter, nx, ny, h, option)

    phi = np.rot90(phi, -1)
    plt.imshow(phi, cmap='hot', origin='lower')
    plt.colorbar()
    plt.xlabel('x/cm')
    plt.ylabel('y/cm')
    plt.show()

if __name__ == "__main__":
    main()
