from numlib import interpolation as inter
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
x=np.array([-5,-4,-3,-2,-1,0,1,2,3,4,5])
y=np.array([25,16,9,4,25,0,1,4,9,2,25])
z=np.linspace(-5,5,100)
fig, ax = plt.subplots()
ax.plot(z, inter.Lagrange(x, y, z), label="Funkcja interpolacyjna")
ax.scatter(x, y, label="Węzły interpolacji")
ax.legend()
plt.show()