import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftshift


def func(x, Lw, U0):
    V = []
    for i in range(len(x)):
        if x[i] < Lw:
            V.append(0)
        else:
            V.append(U0)
    return V


# system variable(take electron for exemple)
hbar = 1.054573 * 10 ** (-34)
m = 9.109 * 10 ** (-31)#mass of electron
e = 1.602 * 10 ** (-19)#charge of elctron
U0 = 2.0#potential divided by e
Lw = 0.9 * 10 ** (-9)
Lb = 0.1 * 10 ** (-9)
a = Lw + Lb
N = 2 **10
fs = 2 * N + 1#sample frequecy
factor = 2 / m / e * (np.pi * hbar / a) ** 2
# print(factor)

x = np.linspace(0, a, fs + 1)[0:-1]
V = func(x, Lw, U0)
# print(len(x))
# print(V)

Vq = fftshift(fft(V))
Vq = np.hstack((np.zeros(N), Vq, np.zeros(N)))/fs #add zeros to Vq and divide fs
# print(Vq)

H = np.zeros((2 * N + 1, 2 * N + 1), dtype="complex_")
for i in range(2 * N + 1):
    H[i, i] = factor * (i - N) ** 2
    for j in range(2 * N + 1):
        H[i, j] += Vq[2 * N + i - j]

energy = np.linalg.eig(H)[0]
Energy = sorted(energy)
for i in range(len(Energy)):
    Energy[i] = Energy[i].real #remove im part

print(Energy[0:3])
x = np.linspace(0, len(Energy), len(Energy))
for i in range(len(Energy)):
    x[i] = x[i] ** 2
plt.scatter(x, Energy)
plt.xlabel('n')
plt.ylabel('E/e')

plt.show()
