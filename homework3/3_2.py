import numpy as np

'''
本程序重新实现了高斯消元法，并用高斯消元法来求解非平衡电桥的等效电阻问题
在函数gaussian(A,b)中我们实现了高斯消元法，首先是判断第i列矩阵元的大小，并使最大的放在第i行
之后再用高斯消元法进行消元。
'''


def gaussian(A, b):
    n = len(A)
    M = np.append(A, np.transpose(b), axis=1)
    # find the max element and swap
    for i in range(n):
        for k in range(i, n):
            if abs(M[k, i]) > abs(M[i, i]):
                M[(i, k), :] = M[(k, i), :]

        # forward
        for j in range(i + 1, n):
            q = M[j, i] / M[i, i]
            M[j] -= q * M[i]

    x = np.zeros(n)
    # backward
    x[n - 1] = float(M[n - 1, n]) / M[n - 1, n - 1]
    for i in range(n - 1, -1, -1):
        z = 0
        for j in range(i + 1, n):
            z = z + float(M[i, j]) * x[j]
        x[i] = float(M[i, n] - z) / M[i, i]

    return x


def main():
    r1 = float(input('请输入r1：'))
    r2 = float(input('请输入r2：'))
    r3 = float(input('请输入r3：'))
    rs = float(input('请输入rs：'))
    ra = float(input('请输入ra：'))
    rx = float(input('请输入rx：'))
    v0 = float(input('请输入v0：'))
    # input the matrices
    A = np.array([[-rx, r1 + ra - rx, -ra], [-r3, -ra, r2 + r3 + ra], [rs + rx + r3, -rx, -r3]], dtype=float)
    b = np.array([[0, 0, v0]], dtype=float)
    # find the solution
    S = gaussian(A, b)
    reff = v0 / S[0]
    print(S)
    print(reff)


if __name__ == '__main__':
    main()
