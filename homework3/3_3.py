import numpy as np
import matplotlib.pyplot as plt

'''
本程序利用高斯基展开求解定态薛定谔方程的基态
程序中所有不带2的物理量表示V（x）=x^2
所有带2的物理量表示V（x）=x^4-x^2
'''


n = 8  # number of bases
S = np.zeros((n + 1, n + 1))
H = np.zeros((n + 1, n + 1))  # initialize S, H matrix
H2 = np.zeros((n + 1, n + 1))
s0 = 0  # default position
v0 = 0.5  # default width of the Gaussian bases
s = np.arange(-n / 2, n / 2 + 1)

# use Gaussian bases with the same width a0
for i in range(len(S)):
    for j in range(len(S[0])):
        S[i, j] = np.sqrt(v0) * np.exp(-0.5 * v0 * (s[i] - s[j]) ** 2) / np.sqrt(2 * np.pi)
        H[i, j] = np.exp(-0.5 * v0 * (s[i] - s[j]) ** 2) * (1 + 2 * v0 ** 2 - 2 * v0 ** 3 * (s[i] - s[j]) ** 2 + \
                                                            v0 * (s[i] + s[j]) ** 2) / (4 * np.sqrt(v0 * 2 * np.pi))
        H2[i, j] = np.exp(-0.5 * v0 * (s[i] - s[j]) ** 2) * (
                3 + 2 * v0 * (3 * (s[i] + s[j]) ** 2 - 2) + (s[i] + s[j]) ** 2 * v0 ** 2 * ((s[i] + s[j]) ** 2 - 4) + \
                8 * v0 ** 3 - 8 * (s[i] - s[j]) ** 2 * v0 ** 4) / (16 * (2 * np.pi) ** 0.5 * v0 ** 1.5)

S_invH = np.matmul(np.linalg.inv(S), H)
energy = np.linalg.eig(S_invH)[0]
Energy = sorted(energy)
index = [np.argwhere(energy == Energy[i])[0][0] for i in range(len(Energy))]
# find the index of the corresponding eigenvector in the sorted Energy
print(Energy)

S_invH2 = np.matmul(np.linalg.inv(S), H2)
energy2 = np.linalg.eig(S_invH2)[0]
Energy2 = sorted(energy2)
index2 = [np.argwhere(energy2 == Energy2[i])[0][0] for i in range(len(Energy2))]
# find the index of the corresponding eigenvector in the sorted Energy
print(Energy2)

x = np.arange(-5, 5, 0.1)
y = []
y2 = []
for j in index:
    dots = []
    dots2 =[]
    for xi in x:
        basis = np.sqrt(v0 / np.pi) * np.exp(-v0 * (xi - s) ** 2)
        eigenvector = (np.linalg.eig(S_invH)[1]).T[j]
        eigenvector2 = (np.linalg.eig(S_invH2)[1]).T[j]
        dots.append(sum(basis * eigenvector))
        dots2.append(sum(basis * eigenvector2))
    y.append(dots)
    y2.append(dots2)

fig = plt.figure(figsize=(14, 7))
fig.add_subplot(1, 2, 1)
plt.title('V(x)=x^2')
for i in range(len(index)):
    plt.plot(x, y[i])

fig.add_subplot(1, 2, 2)
plt.title('V(x)=x^4-x^2')
for i in range(len(index)):
    plt.plot(x,y2[i])

plt.show()
