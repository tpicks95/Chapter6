import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#Load the raw data file
#raw_df = pd.read_excel(r'H:\Documents\CHEM ENG\MACHINE LEARNING\Bayesian Optimisation\EtOH_LAMV_DoE.xlsx')
from main import raw_df
raw_df_sel = raw_df[["SS", "Isothermal Temp", "Mean CB Growth Rate fit to all (um/s)"]]
dropna_raw_df = raw_df_sel.dropna()

#Process conditions (x-variables)
T = dropna_raw_df["Isothermal Temp"]
T = T + 273.15

#Domain y-values
y_GrwRate = dropna_raw_df["Mean CB Growth Rate fit to all (um/s)"]

#Creating domain knowledge plots - IndTime wrt SS
x = T/100
def func(x, Gc, Gd):
    return Gc*x + Gd
xdata = np.ravel(x)
ydata = np.ravel(y_GrwRate)
plt.scatter(xdata, ydata, label = 'data')
popt, _ = curve_fit(func, xdata, ydata)
Gc, Gd = popt
x_line = np.arange(min(x), max(x), 0.001)
plt.plot(x_line, func(x_line, *popt), 'r-', label='fit: Gc=%5.3f, Gd=%5.3f' % tuple(popt))
plt.xlabel("T /K/100")
plt.ylabel("Growth rate /ums-1")
plt.legend()
plt.show()

#Set the target parameters
from main import Target_GR
GrwRate_Target_ums = Target_GR

#Creating y-values for a wider range of x-data from the analytical model above
T = np.arange(0, 6, 0.01)
y_SIM = Gc*T + Gd
y_SIM_Diff = abs(GrwRate_Target_ums - y_SIM)
plt.plot(T, y_SIM_Diff, label = "Analytical model of domain knowledge (mathematically transformed)")
'''def func(x, Gd, Ge, Gf):
    return Gd*x**2 + Ge*x + Gf
popt, _ = curve_fit(func, T, y_SIM_Diff)
Gd, Ge, Gf = popt
plt.plot(T, func(T, *popt), 'r-', label='fit: Gd=%5.3f, Ge=%5.3f, Gf=%5.3f' % tuple(popt))'''
plt.xlabel("Temperature /K/100")
plt.ylabel("Difference between target growth rate and measured growth rate /ums-1")
plt.legend()
plt.show()


