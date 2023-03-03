def Lagrange(x,y,z):
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