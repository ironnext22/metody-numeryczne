def bs(f,a,b):
    iter = 100
    eps = 10e-6
    for k in range(1,iter):
        x=(a+b)/2
        if abs(f(x))<eps:
            break
        else:
            if f(x)*f(a)<0:
                b=x
            else:
                a=x
    return x

def fl(f,a,b):
    iter = 100
    eps = 10e-6
    fa = f(a)
    fb = f(b)
    x1 = a
    x0 = b
    if (fa*fb)>0:
        raise ValueError("Niepoprawny przedział. Funkcja musi mieć różne znaki na krańcach przedziału.")
    for i in range(iter):
        x = (a * fb - b * fa) / (fb - fa)
        fx = f(x)

        if abs(fx) < eps:
            return x

        if fa * fx < 0:
            b = x
            fb = fx
        else:
            a = x
            fa = fx

    return None

def sieczne(f,a,b):
    iter = 100
    eps = 10e-6
    x0=a
    x1=b
    for i in range(iter):
        fx0 = f(x0)
        fx1 = f(x1)
        x2 = x1 - (fx1 * (x1 - x0)) / (fx1 - fx0)

        if abs(x2 - x1) < eps:
            return x2

        x0 = x1
        x1 = x2

    return None

def Newton(f,df,a):
    iter = 100
    eps = 10e-6
    x = a
    for i in range(iter):
        fv = f(x)
        dfv = df(x)
        if abs(fv)<eps:
            return x
        if dfv == 0:
            return None
        x = x-fv/dfv
    return None
import math

# Tutaj definiujemy funkcję, której pierwiastek jest wyliczany
#-------------------------------------------------------------

def f(x):
    return math.sin(x*x-x+1/3.0)+0.5*x

# Tutaj definiujemy pochodną funkcji
#-----------------------------------

def df(x):
    return math.cos(x*x-x+1/3.0)*(2*x-1)+0.5

# Tutaj definiujemy parametry początkowe

def Newtonv2(f,df,x0,n):
    epsx = 1e-14 # Dokładność wyznaczania pierwiastka
    epsy = 1e-14 # Dokładność wyznaczania zera

    # Program główny
    #---------------
    # Zmienne
    f0 = 0
    f1 = 0
    x1 = 0
    result = False

    print("Obliczanie przybliżonego pierwiastka funkcji metodą Newtona")
    print("-----------------------------------------------------------\n")

    while n > 0:
        n -= 1

        # Obliczamy wartość funkcji w punkcie x0
        f0 = f(x0)

        # Sprawdzamy, czy funkcja jest dostatecznie bliska zeru
        if abs(f0) < epsy:
            result = True
            break

        # Obliczamy wartość pierwszej pochodnej funkcji
        f1 = df(x0)

        # Zapamiętujemy bieżące przybliżenie
        x1 = x0

        # Obliczamy kolejne przybliżenie
        x0 -= f0/f1

        # Sprawdzamy, czy odległość pomiędzy dwoma ostatnimi przybliżeniami
        # jest mniejsza od założonej dokładności
        if abs(x1 - x0) < epsx:
            result = True
            break

        # Kontynuujemy obliczenia

    if not result:
        print("Zakończono z błędem!\n")

    print(f"Pierwiastek        x0 = {x0:.15f}")
    print(f"Wartość funkcji f(x0) = {f0:.15f}")
    print(f"Dokładność dla x epsx = {epsx:.15f}")
    print(f"Dokładność dla y epsy = {epsy:.15f}")
    print(f"Liczba obiegów      n = {64 - n}\n")
    return 0