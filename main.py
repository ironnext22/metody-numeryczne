from __future__ import with_statement
from numlib import interpolation as inter
from matplotlib import pyplot as plt
import numpy as np
from numlib import horner as horner
import pandas as pd
from numlib import kwadrat as kwad
from numlib import linear as lin

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
# a = 2
# b = 4
# n = 6
#
# x = [a + i * (b-a)/n for i in range(n+1)]
# def f(x):
#     return (x**2)-5
# print(f"zadanie 1: {kwad.trap(f,x,n)}")
# print(f"zadanie 2a: {kwad.NCtrap(f,a,b,n)}")
# print(f"Zadanie 2b: {kwad.NCSim(f,a,b,n)}")
# var = [0.4,5,-6,-2,55]
# print(f"Zadanie 3: {kwad.NCtrap_wiel(var,-2,2,80,4)}")
# def f2(x):
#     return (x**2)*(np.sin(x)**3)
# a=0
# b=4.5
# n=100
# print(f"Zadanie 4: {kwad.NCtrap(f2,a,b,n)}")
# def f3(x):
#     return np.exp(x**2)*(x-1)
# a=-2
# b=2
# n=100
# print(f"Zadanie 5: {kwad.NCSim(f3,a,b,n)}")
# print(f"gaus test: {kwad.gaus_v1(f3,a,b,n)}")
# def f4(x):
#     return (3*x**3)+(2*x**2)+(8*x)-4
# print(f"Test 1: {kwad.NCtrap(f4,-2,2,100)}")
# def f5(x):
#     return x**x
# print(f"Test 2: {kwad.NCtrap(f5,0,1,10)}")
# def f6(x):
#     return np.sin(1/(1-x))
# print(f"Test 3: {kwad.NCtrap(f6,0,1-np.e-4,1000)}")
# def f7(x):
#     return np.sin(x)/x
# print(f"Test 4: {kwad.NCtrap(f7,0.0001,1,10)}")

# def f10(x):
#     return (x**2)-5
# a = 2
# b = 4
# print(f"zadanie 1a: {kwad.gaus_v1(f10,a,b,2)}")
# print(f"zadanie 1b: {kwad.gaus_v1(f10,a,b,3)}")
# print(f"zadanie 1c: {kwad.gaus_v1(f10,a,b,4)}")
# print(f"zadanie 1d: {kwad.gaus_v1(f10,a,b,5)}")
# def f11(x):
#     return (x**2)*(np.sin(x)**3)
# print(f"zadanie 2a: {kwad.gaus_v1(f11,0,4.5,10000)}")
# def f12(x):
#     return np.exp(x**2)*(x-1)
# print(f"zadanie 2b: {kwad.gaus_v1(f12,-2,2,10000)}")
# def f13(x):
#     return 55-2*x-6*(x**2)+5*(x**3)+0.4*(x**4)
# print(f"test 1: {kwad.gaus_v1(f13,-2.,2.,10000)}")
# def f14(x):
#     return 1-2*np.exp(x)-6*np.cos(x)
# print(f"test 2: {kwad.gaus_v1(f14,-2,2,10000)}")
# def f15(x):
#     return x**2*(np.sin(x)**3)
# print(f"test 3: {kwad.gaus_v1(f15,0,4.5,10000)}")
# print(f"test 4: {kwad.gaus_v1(f12,-2,2,10000)}")
# def f16(x):
#     return np.sin(1-x)/(1-x)
# print(f"test 5: {kwad.gaus_v1(f16,0,1,1000)}")
# print(f"Test 6: {kwad.gaus_v1(f6,0,1-np.e-4,10000)}")
# def f17(x):
#     return np.sin(x)/x
# print(f"Test 7: {kwad.gaus_v1(f17,0,1,10000)}")

A = np.array([[2., 1., -1.], [4., 5., -3.], [2., 8., 7.]])
b = np.array([1., -3., 2.])
x = lin.gauss_elimination(A, b)
A = np.array([[2., 1., -1.], [4., 5., -3.], [2., 8., 7.]])
b = np.array([1., -3., 2.])
y = lin.gauss_crout_elimination(A,b)
# print(x)
# print(y)
A=np.array([[1.,-3.,4,6.8,9.],[-3.,2.,4.6,6.3,-10.],[2.,-1.,2.8,-8.4,-5.],[-6.,2.,7.,-0.5,-0.9],[5.,-2.,-0.5,12.,-4.]])
b=np.array([74.64,-40.26,-2.32,12.6,-8.9])
x1=lin.gauss_elimination(A,b)
A=np.array([[1.,-3.,4,6.8,9.],[-3.,2.,4.6,6.3,-10.],[2.,-1.,2.8,-8.4,-5.],[-6.,2.,7.,-0.5,-0.9],[5.,-2.,-0.5,12.,-4.]])
b=np.array([74.64,-40.26,-2.32,12.6,-8.9])
x2=lin.gauss_crout_elimination(A,b)
print(x1)
print(x2)