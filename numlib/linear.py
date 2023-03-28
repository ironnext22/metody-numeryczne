import numpy as np

#Ax=B
def gauss_elimination(A, b,z=2):
    n = len(b)
    # Elimination phase
    for i in range(n-1):
        for j in range(i+1, n):
            factor = A[j,i] / A[i,i]
            A[j,i+1:n] -= factor * A[i,i+1:n]
            b[j] -= factor * b[i]
    # Back substitution phase
    x = np.zeros(n)
    x[n-1] = round(b[n-1] / A[n-1,n-1],z)
    for i in range(n-2, -1, -1):
        x[i] = round((b[i] - np.dot(A[i,i+1:n], x[i+1:n])) / A[i,i],z)
    return x

def gauss_crout_elimination(A, b,z=2):
    n = len(b)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    # Dekompozycja Crouta
    for i in range(n):
        L[i, i] = 1
        for j in range(i, n):
            sum = 0
            for k in range(i):
                sum += L[i, k] * U[k, j]
            U[i, j] = A[i, j] - sum
        for j in range(i + 1, n):
            sum = 0
            for k in range(i):
                sum += L[j, k] * U[k, i]
            L[j, i] = (A[j, i] - sum) / U[i, i]

    # Rozwiązanie układu równań liniowych
    y = np.zeros(n)
    for i in range(n):
        sum = 0
        for j in range(i):
            sum += L[i, j] * y[j]
        y[i] = (b[i] - sum) / L[i, i]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        sum = 0
        for j in range(i + 1, n):
            sum += U[i, j] * x[j]
        x[i] = round((y[i] - sum) / U[i, i],z)

    return x