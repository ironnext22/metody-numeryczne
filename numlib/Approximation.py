import numpy as np
from numlib import linear as l
def ap(X,Y,n):
  A = np.zeros((n+1,n+1))
  B = np.zeros(n+1)
  for i in range(n+1):
    for j in range(n+1):
      for k in range(len(X)):
        A[i][j] += X[k]**(i+j)

    for j in range(len(X)):
      B[i] += Y[k]*(X[k]**i)
  W = l.gauss_elimination(A,B)
  return W

