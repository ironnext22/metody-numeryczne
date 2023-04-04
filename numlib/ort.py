import numpy as np
from numlib import kwadrat as kwa

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

