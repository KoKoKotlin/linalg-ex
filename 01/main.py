import random
import copy

import numpy as np
from numpy.lib.function_base import digitize

import scipy
from scipy.sparse import diags
from scipy.linalg import lu

from matplotlib import pyplot as plt

def scalar_mul(a, b):
    assert a.shape == b.shape, "Vectors are not of the same size!"

    res = 0
    for i in range(a.shape[0]):
        res += a[i] * b[i]

    return res

def ex01():
    # a & b
    a = np.array([i for i in range(10)])
    b = np.array([i % 3 for i in range(10)])
    print(a, a.shape, b, b.shape)
    print("a.T * b == ", a.T @ b, "==", scalar_mul(a, b))
    print("a * b.T == ", a @ b.T)

    # c
    n = 10
    c = np.array([np.random.rand() for _ in range(n)])
    d = np.array([random.gauss(0, 1) for _ in range(n)])
    print("c * d.T ==", c.T @ d)

    # d
    plt.hist(np.hstack([c, d]), bins=10)
    plt.show()

def compute_LU(A):
    n = A.shape[0]

    L = np.diag([1.0] * n)
    U = copy.copy(A)

    for pivot in range(0, n - 1):
        for row in range(pivot + 1, n):
            factor = U[row, pivot] / U[pivot, pivot]
            L[row, pivot] = factor

            for i in range(n):
                U[row, i] -= factor * U[pivot, i]

    return L, U

def solve_system(A, b, L, U):
    assert A.shape[0] == b.shape[0], "Matrix and vector are not compatible!"

    n = A.shape[0]
    sol = np.array([0.0] * n)

    sol[0] = b[0]
    sol[n - 1] = b[n - 1]

    # L * y = b
    y = np.array([0.0] * n)
    y[0] = b[0]
    y[n - 1] = b[n - 1]
    for i in range(1, n - 1):
        y[i] = b[i]
        for j in range(i):
            y[i] -= L[j, i] * y[j]

    # U * x = y
    x = np.array([0.0] * n)
    for i in range(n - 1, 0, -1):
        factor = 1.0 / U[i, i]
        x[i] = y[i]

        for j in range(n - 1, i, -1):
            x[i] -= U[j, i] * y[j]

        x[i] *= factor

    return x

def ex02():
    # b
    # for loops
    n = 3
    A = np.array([[0.0] * n for _ in range(n)])
    for i in range(n):
        if i - 1 >= 0:
            A[i - 1, i] = -1.0

        A[i, i] = 2

        if i + 1 < n:
            A[i + 1, i] = -1.0
    print(A)

    # scipy
    B = diags([[-1.0] * (n - 1), [2.0] * n, [-1.0] * (n - 1)], [-1, 0, 1])
    print(B)

    L, U = compute_LU(A)

    print(L)
    print(U)

    LU = lu(A)
    print(LU[0])
    print(LU[1])

    # c
    A[:, 0] = A[0, :] = 0
    A[:, n - 1] = A[n - 1, :] = 0
    A[0, 0] = 1
    A[n - 1, n - 1] = 1
    print(A)

    b = 1.0/(n-1.0)**2.0 * np.array([4.0] * n)
    print(b)

    numpy_sol = np.linalg.solve(A, b)
    scipy_sol = scipy.linalg.solve(A, b)
    x = solve_system(A, b, L, U)

    print(numpy_sol, scipy_sol, x)

    # d

if __name__ == "__main__":
    ex02()