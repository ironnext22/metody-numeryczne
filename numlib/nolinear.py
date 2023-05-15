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