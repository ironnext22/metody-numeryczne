import numpy as np
#zadanie 1
def horner_natural(var,n,x):
    if n == 0 :
        return var[0]
    elif n < 0 :
        raise ValueError("Error")

    return x*horner_natural(var,n-1,x)+var[n]
#zadanie 2
def newton_horner(x,xi,b):
    n = len(xi)-1
    w=b[0]
    for i in range(n,1,1):
        w=w*(x-xi[i])+b[i]
    return w
#zadanie 3
def newton_to_natural(x, y):
    n = len(x)
    if n != len(y):
        raise ValueError("Lista x i y muszą mieć taką samą długość")
    # Inicjalizacja tablicy różnic dzielonych
    F = [[0 for i in range(n)] for j in range(n)]
    for i in range(n):
        F[i][0] = y[i]

    for j in range(1, n):
        for i in range(n - j):
            F[i][j] = (F[i + 1][j - 1] - F[i][j - 1]) / (x[i + j] - x[i])
    return F[0][::-1]