def Lagrange(x,y,z): #x-węzły interpolacyjne, y-f(x)
    n=len(x)
    if n!= len(y):
        raise ValueError("Lista x i y muszą mieć taką samą długość")
    s=0
    for i in range(n):
        p=1
        for j in range(n):
            if j != i:
                p *= (z-x[j])/(x[i]-x[j])
        s += y[i]*p
    return s
def newton_interpolation(x, y, z):
    def divided_diff(x, y):
        n = len(x)
        if n != len(y):
            raise ValueError("Lista x i y muszą mieć taką samą długość")
        # Inicjalizacja tablicy różnic dzielonych
        F = [[0 for i in range(n)] for j in range(n)]
        for i in range(n):
            F[i][0] = y[i]

        # Obliczanie różnic dzielonych
        for j in range(1, n):
            for i in range(n - j):
                F[i][j] = (F[i + 1][j - 1] - F[i][j - 1]) / (x[i + j] - x[i])

        return F[0]
    n = len(x)
    if n != len(y):
        raise ValueError("Lista x i y muszą mieć taką samą długość")

    F = divided_diff(x, y)
    p = F[n-1]
    for k in range(1, n):
        p = F[n-1-k] + (z - x[n-1-k])*p

    return p