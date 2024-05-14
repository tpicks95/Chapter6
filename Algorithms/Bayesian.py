#Import Modules

#GPyOpt - Cases are important, for some reason
import GPyOpt
from GPyOpt.methods import BayesianOptimization

#numpy
import numpy as np
from numpy.random import multivariate_normal #For later example

import pandas as pd

#Plotting tools
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from numpy.random import multivariate_normal

from INDTIMEwrtSS_SurrogateModel import Ia, Ib
from GRWRATEwrtSS_SurrogateModel import Ga, Gb
from GRWRATEwrtT_SurrogateModel import Gc, Gd
from NUCRATEwrtSS_SurrogateModel import Na, Nb
from NUCRATEwrtT_SurrogateModel import Nc, Nd

#Plotting the 3D representation of the objective function
def obj_func_2d(x,y):
    from INDTIMEwrtSS_SurrogateModel import IndTime_Target_s
    y_SIM_IndwrtSS = Ia * np.exp(-Ib * x)
    y_SIM_Diff_IndwrtSS = abs(np.log(IndTime_Target_s) - np.log(y_SIM_IndwrtSS))

    from GRWRATEwrtSS_SurrogateModel import GrwRate_Target_ums
    y_SIM_GrwRatewrtSS = Ga * x + Gb
    y_SIM_Diff_GrwRatewrtSS = abs(GrwRate_Target_ums - y_SIM_GrwRatewrtSS)

    from GRWRATEwrtT_SurrogateModel import GrwRate_Target_ums
    y_SIM_GrwRatewrtT = Gc * y + Gd
    y_SIM_Diff_GrwRatewrtT = abs(GrwRate_Target_ums - y_SIM_GrwRatewrtT)

    from NUCRATEwrtSS_SurrogateModel import NucRate_Target_cps
    y_SIM_NucRatewrtSS = Na * np.exp(Nb * x)
    y_SIM_Diff_NucRatewrtSS = abs(np.log(NucRate_Target_cps) - np.log(y_SIM_NucRatewrtSS))

    from NUCRATEwrtT_SurrogateModel import NucRate_Target_cps
    y_SIM_NucRatewrtT = Nc * y + Nd
    y_SIM_Diff_NucRatewrtT = abs(NucRate_Target_cps - y_SIM_NucRatewrtT)

    IndTime = y_SIM_Diff_IndwrtSS
    GrwRate = y_SIM_Diff_GrwRatewrtSS + y_SIM_Diff_GrwRatewrtT
    NucRate = y_SIM_Diff_NucRatewrtSS + y_SIM_Diff_NucRatewrtT
    return IndTime + 10 * GrwRate + NucRate  # Growth rate has 10x factor due to being very small numbers/ magnitude out of normalisation compared to the others

fig = plt.figure(figsize=plt.figaspect(0.3))
fig.suptitle('Plots of our objective function')

# Make data.
X = np.arange(1, 4, 0.01)
Y = np.arange(1, 4, 0.01)
X, Y = np.meshgrid(X, Y)
Z = obj_func_2d(X,Y)

# First subplot
ax = fig.add_subplot(1, 2, 1)
ax.contour(X,Y,Z)

# Second subplot
ax = fig.add_subplot(1, 2, 2, projection='3d')

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()


#Running the optimisation
def objfunc2d(x):
    """
    x is a 2 dimensional vector.
    """
    x1 = x[:, 0]
    x2 = x[:, 1]
    from INDTIMEwrtSS_SurrogateModel import IndTime_Target_s
    y_SIM_IndwrtSS = Ia * np.exp(-Ib * x1)
    y_SIM_Diff_IndwrtSS = abs(np.log(IndTime_Target_s) - np.log(y_SIM_IndwrtSS))

    from GRWRATEwrtSS_SurrogateModel import GrwRate_Target_ums
    y_SIM_GrwRatewrtSS = Ga * x1 + Gb
    y_SIM_Diff_GrwRatewrtSS = abs(GrwRate_Target_ums - y_SIM_GrwRatewrtSS)

    from GRWRATEwrtT_SurrogateModel import GrwRate_Target_ums
    y_SIM_GrwRatewrtT = Gc * x2 + Gd
    y_SIM_Diff_GrwRatewrtT = abs(GrwRate_Target_ums - y_SIM_GrwRatewrtT)

    from NUCRATEwrtSS_SurrogateModel import NucRate_Target_cps
    y_SIM_NucRatewrtSS = Na * np.exp(Nb * x1)
    y_SIM_Diff_NucRatewrtSS = abs(np.log(NucRate_Target_cps) - np.log(y_SIM_NucRatewrtSS))

    from NUCRATEwrtT_SurrogateModel import NucRate_Target_cps
    y_SIM_NucRatewrtT = Nc * x2 + Nd
    y_SIM_Diff_NucRatewrtT = abs(NucRate_Target_cps - y_SIM_NucRatewrtT)

    IndTime = y_SIM_Diff_IndwrtSS
    GrwRate = y_SIM_Diff_GrwRatewrtSS + y_SIM_Diff_GrwRatewrtT
    NucRate = y_SIM_Diff_NucRatewrtSS + y_SIM_Diff_NucRatewrtT
    return IndTime + 10 * GrwRate + NucRate  # Growth rate has 10x factor due to being very small numbers/ magnitude out of normalisation compared to the others

bounds2d = [{'name': 'var_1', 'type': 'continuous', 'domain': (1,4)},
            {'name': 'var_2', 'type': 'continuous', 'domain': (2.7,3.3)}]
maxiter = 100

myBopt_2d = GPyOpt.methods.BayesianOptimization(objfunc2d,
                                                domain=bounds2d,
                                                model_type='GP',
                                                acquisition_type ='EI',
                                                acquisition_jitter = 1,
                                                exact_feval=True,
                                                verbosity=False,
                                                verbosity_model=True)
myBopt_2d.run_optimization(max_iter = maxiter)
print("="*20)
print("Value of (x,y) that minimises the objective:"+str(myBopt_2d.x_opt))
print("Supersaturation of:")
print((myBopt_2d.x_opt[0]))
print("Temperature of:")
print(((myBopt_2d.x_opt[1])*100)-273.15)
print("Minimum value of the objective: "+str(myBopt_2d.fx_opt))
print("="*20)
myBopt_2d.plot_acquisition()


#Plot some more characteristics:
myBopt_2d.plot_convergence() #Can clearly see it spends quite some time exploring the best small section which it thinks is the best space