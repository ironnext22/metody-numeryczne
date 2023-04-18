import numpy as np
from numlib import kwadrat as kwd
import math
import scipy
def gram_schmidt(A):
    # Rozmiar macierzy A
    m, n = A.shape
    # Tworzymy nową macierz Q i inicjujemy ją zerami
    Q = np.zeros((m,n))
    # Iterujemy po kolumnach macierzy A
    for j in range(n):
        # Kolumna j z macierzy A
        v = A[:, j]
        # Iterujemy po już wyznaczonych wektorach Q
        for i in range(j):
            # Wektor i z macierzy Q
            q = Q[:, i]
            # Wyznaczamy nową wartość wektora v

            v = v - np.dot(v, q) * q
        # Normalizujemy wektor v i dodajemy go do macierzy Q
        Q[:, j] = v / (sum([xi**2 for xi in v])**0.5)
    # Wyznaczamy macierz R
    return Q

def iloczyn(A,B,a,b,n):
    n1 = len(A)
    m = len(B)

    s = n1+m-1
    wynik = np.zeros(s)
    for i in range(n1):
        for j in range(m):
            wynik[j+i] += A[i]*B[j]

    return kwd.NCSim_wiel(wynik,a,b,n)

def generate_basis_standard(n):
    basis = np.zeros((n, n))
    for i in range(n):
        basis[i][n-i-1] = 1
    return basis

def gram_schmidtv2(A, a, b, n):
    wynik = []

    for i in range(len(A)):
        w = []

        for j in range(len(A[i])):
            pom = A[i][j]

            for k in range(i):
                s = round(iloczyn(A[i], wynik[k],a,b,n) / iloczyn(wynik[k], wynik[k],a,b,n),6)
                pom -= s * wynik[k][j]

            w.append(pom)

        wynik.append(w)
        w2 = np.array(wynik)
    return w2


def generate_orthogonal_basis(n, a, b, weight_func = 1):
    # Generate nodes and weights using quadrature
    nodes, weights = np.polynomial.legendre.leggauss(n)

    # Define basis functions
    basis_functions = []
    for i in range(n):
        def basis_function(x, i=i):
            return np.sqrt(weights[i]) * weight_func(x) * np.polynomial.legendre.legval(x, [0] * i + [1] + [0] * (
                        n - i - 1))

        basis_functions.append(basis_function)

    # Generate orthogonal matrix
    orthogonal_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            integral = scipy.integrate.quad(lambda x: basis_functions[i](x) * basis_functions[j](x), a, b)
            orthogonal_matrix[i][j] = integral[0]

    # Calculate Cholesky decomposition of orthogonal matrix
    L = np.linalg.cholesky(orthogonal_matrix)

    # Define orthonormal basis functions
    orthonormal_basis_functions = []
    for i in range(n):
        def orthonormal_basis_function(x, i=i):
            return (1 / np.sqrt(b - a)) * np.dot(L[i], [basis_function(x) for basis_function in basis_functions])

        orthonormal_basis_functions.append(orthonormal_basis_function)

    # Return orthonormal basis functions
    return orthonormal_basis_functions