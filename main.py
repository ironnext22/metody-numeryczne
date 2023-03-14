from numlib import interpolation as inter
from matplotlib import pyplot as plt
import numpy as np
from numlib import horner as horner
import pandas as pd
from numlib import kwadrat as kwad

# var=np.array([1/2,4/3,-13/6,-2])
# x=-4
# wynik = horner.horner_natural(var,3,x)
# print(f"Wynik dla {x} wynosi: {wynik}")
#
# xi = np.array([0,-4,-1,0])
# b=np.array([-4,5/3,-7/6,1/2])
# wynik=horner.newton_horner(x,xi,b)
# print(f"Wynik dla {x} wynosi: {wynik}")
#
# x = np.array([-1, 0, 1])
# f = np.array([2, -1, 0])
# wynik = horner.newton_to_natural(x,f)
# print(f"Otrzymany wielomian: {wynik}")
#
# x=np.array([-2,-1,0,1,2])
# y=np.array([5,-2,4,-7,2])
# z=np.linspace(-2,2,100)
# fig, ax = plt.subplots()
# ax.plot(z, inter.Lagrange(x, y, z), label="Funkcja interpolacyjna Lagranga")
# ax.scatter(x, y, label="Węzły interpolacji 1")
# plt.title("interpolacja Lagranga")
# ax.legend()
# plt.show()
#
# x=np.array([-2,-1,0,1,2])
# y=np.array([5,-2,4,-7,2])
# z=np.linspace(-2,2,100)
# fig, ax = plt.subplots()
# ax.plot(z, inter.newton_interpolation(x, y, z), label="Funkcja interpolacyjna Lagranga")
# ax.scatter(x, y, label="Węzły interpolacji 1")
# plt.title(" Postać Newtona wielomianu Lagrange’a")
# ax.legend()
# plt.show()
a = 2
b = 4
n = 6

x = [a + i * (b-a)/n for i in range(n+1)]
def f(x):
    return (x**2)-5
print("zadanie 1")
print(kwad.trap(f,x,n))
print(kwad.met_trapv2(f,a,b,n))

