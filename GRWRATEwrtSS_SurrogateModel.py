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
SS = dropna_raw_df["SS"]

#Domain y-values
y_GrwRate = dropna_raw_df["Mean CB Growth Rate fit to all (um/s)"]

#Creating domain knowledge plots - IndTime wrt SS
x = SS
def func(x, Ga, Gb):
    return Ga*x + Gb
xdata = np.ravel(x)
ydata = np.ravel(y_GrwRate)
plt.scatter(xdata, ydata, label = 'data')
popt, _ = curve_fit(func, xdata, ydata)
Ga, Gb = popt
x_line = np.arange(min(x), max(x), 0.001)
plt.plot(x_line, func(x_line, *popt), 'r-', label='fit: Ga=%5.3f, Gb=%5.3f' % tuple(popt))
plt.xlabel("SS")
plt.ylabel("Growth rate /ums-1")
plt.legend()
plt.show()

#Set the target parameters
from main import Target_GR
GrwRate_Target_ums = Target_GR

#Creating y-values for a wider range of x-data from the analytical model above
SS = np.arange(0, 5, 0.01)
y_SIM = Ga*SS + Gb
y_SIM_Diff = abs(GrwRate_Target_ums - y_SIM)
plt.plot(SS, y_SIM_Diff, label = "Analytical model of domain knowledge (mathematically transformed)")
'''def func(x, Ga, Gb, Gc):
    return Ga*x**2 + Gb*x + Gc
popt, _ = curve_fit(func, SS, y_SIM_Diff)
Ga, Gb, Gc = popt
plt.plot(SS, func(SS, *popt), 'r-', label='fit: Ga=%5.3f, Gb=%5.3f, Gc=%5.3f' % tuple(popt))'''
plt.xlabel("SS")
plt.ylabel("Difference between target growth rate and measured growth rate /ums-1")
plt.legend()
plt.show()


