import numpy as np
#zadanie 1
def horner_natural(var,n,x):
    if n == 0 :
        return var[0]
    elif n < 0 :
        raise ValueError("XD")

    return x*horner_natural(var,n-1,x)+var[n]

#zadanie 2
def horner_newton(zeros,x):
    n = len(zeros)
    a = x - zeros[n-1]
    if n-1 > 0 :
        a = a * horner_newton(zeros[:n-1],x)
    return a
#zadanie 3
def newton_natural(zeros):
    n = len(zeros)