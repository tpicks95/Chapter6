import matplotlib.pyplot as plt
import numpy as np


fig = plt.figure() # create new figure
ax = fig.add_subplot(projection='3d') # add 3d plot

#x is logT and y is logSS
n = 5000
x = np.random.rand(n)
y = np.random.rand(n)
z = (Ia+Ga+Na)*x**2 + (Ib+Gb+Nb)*x + (Id+Gd+Nd)*y**2 + (Ie+Ge+Ne)*y + (Ic+Gc+Nc+If+Gf+Nf)

plt.show()

'''from INDTIME_SurrogateModel import Ia, Ib, Ic, Id, Ie, If
from GRWRATE_SurrogateModel import Ga, Gb, Gc, Gd, Ge, Gf
from NUCRATE_SurrogateModel import Na, Nb, Nc, Nd, Ne, Nf'''