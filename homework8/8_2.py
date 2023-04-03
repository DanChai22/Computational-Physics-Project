import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from matplotlib.animation import FuncAnimation
from scipy import sparse


def initial(x):
    k0 = 1
    xi0 = -5
    return np.sqrt(1 / np.pi) * np.exp(complex(-0.5 * (x - xi0) ** 2, k0 * x))

# system parameters
nx = 300
nt = 300
x = np.linspace(-15, 15, nx)
dx = x[1] - x[0]
dt = 0.01
alpha = dt / (dx ** 2)

# initialize
psi = np.array([[complex(0, 0) for i in range(nx)] for j in range(nt)])
for i in range(len(psi[0])):
    psi[0, i] = initial(x[i])


# evaluate Matrix
F = np.zeros(nx)
for i in range(nx):
    if abs(x[i]) <= 3:
        F[i] = 5
F = np.diag(F)
I = np.eye(nx)
B = sparse.diags([-1, 2, -1], [-1, 0, 1], shape=(nx, nt)).toarray()

#evaluate phi
for i in range(1, len(psi)):
    A = (2 * complex(0, -1) * I - dt * F + alpha * B)
    b = np.dot(2 * complex(0, -1) * I + dt * F - alpha * B, psi[i - 1].reshape((-1, 1)))
    b = (b.reshape((1, -1))[0])
    psi[i] = (np.array(linalg.solve(A, b)))


# plot
plt.subplot(1, 3, 1)
plt.plot(x, abs(psi[0]))
plt.title('t=0')
plt.subplot(1, 3, 2)
plt.plot(x, abs(psi[100]))
plt.title('t=1')
plt.subplot(1, 3, 3)
plt.plot(x, abs(psi[200]))
plt.title('t=2')


#  animation
fig, ax = plt.subplots()
ax = plt.axes(xlim=(-10, 10), ylim=(-0.5, 1))
xdata, ydata = [], []
ln, = ax.plot([], [], animated=False)


def update(frame):
    y = abs(psi[frame])
    ln.set_data(x, y)
    return ln,


anim = FuncAnimation(fig, update, range(nt), interval=1, blit=True)
plt.show()
