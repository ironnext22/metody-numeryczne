from __future__ import with_statement
from turtle import pu

import numpy as np
import scipy
from numlib import horner as horner
import decimal
#
import numpy as np
import numpy.linalg as la
import math
from numpy.core.multiarray import normalize_axis_index


#zadanie 1
def trap(f,x,n):
    dx = (x[-1] - x[0]) / n
    s = (f(x[0]) + f(x[-1])) / 2

    for i in range(1, n):
        s += f(x[i])

    return s * dx

def met_trapv2(f,a,b,n):

    x = np.array([0])
    h = (b - a) / n
    x = [a + i * (b-a)/n for i in range(n+1)]

    wynik = (f(a)+f(b))/2
    for i in range(1, n):
        wynik += f(x[i])

    return wynik*h

#zadanie 2a

def NCtrap(f,a,b,n):
    h=(b-a)/n
    x=[a+i*h for i in range(n+1)]
    y=[f(i) for i in x]
    return (h/2)*(y[0]+2*sum(y[1:-1])+y[-1])

def NCtrapv2(f,a,b,h):
    n=int((b-a)/h)
    if n<=0:
        raise ValueError("Error")
    wynik = 0
    x=[a+i*h for i in range(n+1)]
    y=[f(i) for i in x]
    return (h/2)*(y[0]+2*sum(y[1:-1])+y[-1])

#zadanie 2b
def NCSim(f,a,b,n):
    if n%2 == 1:
        raise ValueError("Error")
    h=(b-a)/n
    x=[a+i*h for i in range(n+1)]
    y=[f(i) for i in x]
    wynik = (h/3)*(y[0]+4*sum(y[1:-1:2])+2*sum(y[2:-1:2])+y[-1])

    return wynik

def NCSimv2(f,a,b,h):
    n = int((b-a)/h)
    if n % 2 == 1:
        raise ValueError("Error")

    x=[a+i*h for i in range(n+1)]
    y=[f(i) for i in x]
    wynik = (h/3)*(y[0]+4*sum(y[1:-1:2])+2*sum(y[2:-1:2])+y[-1])
    return wynik



#zadanie 3
def NCtrap_wiel(f,a,b,n,stop):
    h=(b-a)/n
    x=[a+i*h for i in range(n+1)]
    y=[horner.horner_natural(f,stop,i) for i in x]
    return (h/2)*(y[0]+2*sum(y[1:-1])+y[-1])

# def NCSim_wiel(f,a,b,n,stop):
#     if n%2 == 1:
#         raise ValueError("Error")
#     h=(b-a)/n
#     x=[a+i*h for i in range(n+1)]
#     y=[kwad.horner_natural(f,stop,i) for i in x]
#     wynik = (h/3)*(y[0]+4*sum(y[1:-1:2])+2*sum(y[2:-1:2])+y[-1])
#
#     return wynik

def gaus_v1(f,a,b,n):
    x,w=np.polynomial.legendre.leggauss(n)
    return ((b-a)/2)*np.sum(w*f((b-a)/2*x+(a+b)/2))

def gauss_v2(f, a, b, n):
    # Obliczenie węzłów i wag dla metody Gaussa-Legendre'a
    x = [0] * n
    w = [0] * n
    # Wyznaczanie węzłów i wag
    m = (n + 1) // 2
    for i in range(1, m + 1):
        z = math.cos(math.pi * (i - 0.25) / (n + 0.5))
        while True:
            p1 = 1.0
            p2 = 0.0
            for j in range(n):
                p3,p2 = p2,p1
                p1 = ((2.0 * j + 1.0) * z * p2 - j * p3) / (j + 1.0)
            pp = n * (z * p1 - p2) / (z * z - 1.0)
            z1 = z
            z = z1 - p1 / pp
            if abs(z - z1) < 1e-15:
                break
        x[i - 1] = -z
        x[n - i] = z
        w[i - 1] = 2.0 / ((1.0 - z * z) * pp * pp)
        w[n - i] = w[i - 1]
    # Przeksztalcenie przedzialu z [a, b] na [-1, 1]
    t = [0] * n
    for i in range(n):
        t[i] = (b - a) / 2.0 * x[i] + (b + a) / 2.0
    # Obliczenie calki
    integral = 0.0
    for i in range(n):
        integral += w[i] * f(t[i])
    return integral * (b - a) / 2.0


