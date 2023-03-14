import numpy as np
from numlib import horner as kwad
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
    wynik = 0
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
    wynik = 0
    x=[a+i*h for i in range(n+1)]
    y=[kwad.horner_natural(f,stop,i) for i in x]
    return (h/2)*(y[0]+2*sum(y[1:-1])+y[-1])

def NCSim_wiel(f,a,b,n,stop):
    if n%2 == 1:
        raise ValueError("Error")
    h=(b-a)/n
    x=[a+i*h for i in range(n+1)]
    y=[kwad.horner_natural(f,stop,i) for i in x]
    wynik = (h/3)*(y[0]+4*sum(y[1:-1:2])+2*sum(y[2:-1:2])+y[-1])

    return wynik
