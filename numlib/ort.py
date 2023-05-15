import numpy as np
from numlib import kwadrat as kwd
import math
from scipy.integrate import quad
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

def gram_schmidtv2(A, a, b, n):
    wynik = []

    for i in range(len(A)):
        w = []

        for j in range(len(A[i])):
            pom = A[i][j]

            for k in range(i):
                s = round(iloczyn(A[i], wynik[k],a,b,n) / iloczyn(wynik[k], wynik[k],a,b,n),6)
                pom -= round(s * wynik[k][j],6)

            w.append(pom)

        wynik.append(w)
        w2 = np.transpose(wynik)
    return w2


def generate_basis_standard(n):
    basis = np.zeros((n, n))
    for i in range(n):
        basis[i][n-i-1] = 1
    return basis