import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#Set the target parameters
from main import Target_GR
GrwRate_Target_ums = Target_GR
GrwRate_Target_ums_log = math.log10(GrwRate_Target_ums)

#Load the raw data file
#raw_df = pd.read_excel(r'H:\Documents\CHEM ENG\MACHINE LEARNING\Bayesian Optimisation\EtOH_LAMV_DoE.xlsx')
from main import raw_df
raw_df_sel = raw_df[["SS", "Isothermal Temp", "Mean CB Growth Rate fit to all (um/s)"]]
dropna_raw_df = raw_df_sel.dropna()

#Process conditions (x-variables)
SS = dropna_raw_df["SS"]
SS = 10*np.log10(SS)
T = dropna_raw_df["Isothermal Temp"]
#T = (1/(T+273.15))
T = np.log10(T)

#y-values
y_GrwRate = dropna_raw_df["Mean CB Growth Rate fit to all (um/s)"]
y_GrwRate_log = np.log10(y_GrwRate)
y_DiffLogGrwRate = abs(GrwRate_Target_ums_log - y_GrwRate_log)

#Creating plots - IndTime wrt SS
x = SS
def func(x, Ga, Gb, Gc):
    return Ga*x**2 + Gb*x + Gc
xdata = np.ravel(x)
ydata = np.ravel(y_DiffLogGrwRate)
plt.scatter(xdata, ydata, label = 'data')
popt, pcov = curve_fit(func, xdata, ydata)
Ga, Gb, Gc = popt
x_line = np.arange(min(x), max(x), 0.001)
plt.plot(x_line, func(x_line, *popt), 'r-', label='fit: Ga=%5.3f, Gb=%5.3f, Gc=%5.3f' % tuple(popt))
plt.xlabel("log of SS")
plt.ylabel("Difference between target and measured growth rate (logged)")
plt.legend()
plt.show()

#Creating plots - IndTime wrt T
x = T
def func(x, Gd, Ge, Gf):
    return Gd*x**2 + Ge*x + Gf
xdata = np.ravel(x)
ydata = np.ravel(y_DiffLogGrwRate)
plt.scatter(xdata, ydata, label = 'data')
popt, pcov = curve_fit(func, xdata, ydata)
Gd, Ge, Gf = popt
x_line = np.arange(min(x), max(x), 0.001)
plt.plot(x_line, func(x_line, *popt), 'r-', label='fit: Gd=%5.3f, Ge=%5.3f, Gf=%5.3f' % tuple(popt))
plt.xlabel("log of T")
plt.ylabel("Difference between target and measured growth rate (logged)")
plt.legend()
plt.show()