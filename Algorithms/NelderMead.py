from MyProblem import problem
from pymoo.algorithms.soo.nonconvex.nelder import NelderMead
from pymoo.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt

algorithm = NelderMead()

res = minimize(problem,
               algorithm,
               seed=1,
               verbose=False)

print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))

SS = res.X[0]
roundSS =  str(round(SS,2))
T = res.X[1]
T = (T)*100 - 273.15
roundT = str(round(T,2))

print(f"The ideal input conditions are a temperature of {roundT} and a supersaturation of {roundSS}")

#3D Graphing the surface
from INDTIMEwrtSS_SurrogateModel import Ia, Ib
from GRWRATEwrtSS_SurrogateModel import Ga, Gb
from GRWRATEwrtT_SurrogateModel import Gc, Gd
from NUCRATEwrtSS_SurrogateModel import Na, Nb
from NUCRATEwrtT_SurrogateModel import Nc, Nd

x = np.arange(1, 4, 0.01)
y = np.arange(1, 4, 0.01)

# x is SS and y is Temperature
def f(x,y):
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

X, Y = np.meshgrid(x,y)
Z = f(X,Y)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 500, cmap='turbo')
ax.set_zlim(zmin=0)
ax.set_xlabel('SS')
ax.set_ylabel('Temperature /K/100')
ax.set_zlabel('Objective function')
plt.show()