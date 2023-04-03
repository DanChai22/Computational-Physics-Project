# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

n=10

def h(x, y, z, s, vector):  # hamiltonian
    s1 = s[(x + 1) % n, y, z]
    s2 = s[(x - 1) % n, y, z]
    s3 = s[x, (y + 1) % n, z]
    s4 = s[x, (y - 1) % n, z]
    s5 = s[z, y, (z + 1) % n]
    s6 = s[z, y, (z - 1) % n]
    h = np.inner(vector, s1 + s2 + s3 + s4 + s5 + s6)
    return -h


def get_random_vector():
        phi = np.random.random() * 2 * np.pi
        theta = np.arccos(1 - 2 * np.random.random())
        s = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
        return s


    # %%
m = []
T = [i for i in np.arange(1, 2, 0.1)]
for t in T:
    beta = 1 / t
    s = np.array([[[[float(1), float(0), float(0)] for i in range(n)] for j in range(n)] for k in range(n)])

    for time in range(int(n ** 3 * 1000)):
        index = [np.random.randint(0, n), np.random.randint(0, n), np.random.randint(0, n)]
        vector = get_random_vector()

        H = h(index[0], index[1], index[2], s, s[index[0], index[1], index[2]])
        H_modified = h(index[0], index[1], index[2], s, vector)

        if H > H_modified:
            s[index[0], index[1], index[2]] = vector
        else:
            r = np.exp(-beta * (H_modified - H))
            if np.random.random() < r:
                s[index[0], index[1], index[2]] = vector

    m.append(np.linalg.norm(sum(sum(sum(s)))) / n ** 3)

plt.plot(T, m)


# %%

fig = plt.figure()
ax = Axes3D(fig)
x = np.linspace(0, n - 1, n)
y = np.linspace(0, n - 1, n)
z = np.linspace(0, n - 1, n)
x, y, z = np.meshgrid(x, y, z)

sx = [[[s[i, j, k][0] for k in range(n)] for j in range(n)] for i in range(n)]
sy = [[[s[i, j, k][1] for k in range(n)] for j in range(n)] for i in range(n)]
sz = [[[s[i, j, k][2] for k in range(n)] for j in range(n)] for i in range(n)]

ax.quiver(x, y, z, sx, sy, sz, normalize=True)

plt.show()

