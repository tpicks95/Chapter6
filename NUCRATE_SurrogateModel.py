import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#Set the target parameters
from main import Target_NR
NucRate_Target_cps = Target_NR
NucRate_Target_cps_log = math.log10(NucRate_Target_cps)

#Load the raw data file
#raw_df = pd.read_excel(r'H:\Documents\CHEM ENG\MACHINE LEARNING\Bayesian Optimisation\EtOH_LAMV_DoE.xlsx')
from main import raw_df
raw_df_sel = raw_df[["SS", "Isothermal Temp", "Mean CB Nucleation Rate (#/s)"]]
dropna_raw_df = raw_df_sel.dropna()

#Process conditions (x-variables)
SS = dropna_raw_df["SS"]
SS = 10*np.log10(SS)
T = dropna_raw_df["Isothermal Temp"]
#T = (1/(T+273.15))
T = np.log10(T)

#y-values
y_NucRate = dropna_raw_df["Mean CB Nucleation Rate (#/s)"]
y_NucRate_log = np.log10(y_NucRate)
y_DiffLogNucRate = abs(NucRate_Target_cps_log - y_NucRate_log)

#Creating plots - IndTime wrt SS
x = SS
def func(x, Na, Nb, Nc):
    return Na*x**2 + Nb*x + Nc
xdata = np.ravel(x)
ydata = np.ravel(y_DiffLogNucRate)
plt.scatter(xdata, ydata, label = 'data')
popt, pcov = curve_fit(func, xdata, ydata)
Na, Nb, Nc = popt
x_line = np.arange(min(x), max(x), 0.001)
plt.plot(x_line, func(x_line, *popt), 'r-', label='fit: Na=%5.3f, Nb=%5.3f, Nc=%5.3f' % tuple(popt))
plt.xlabel("log of SS")
plt.ylabel("Difference between target and measured nucleation rate (logged)")
plt.legend()
plt.show()

#Creating plots - IndTime wrt T
x = T
def func(x, Nd, Ne, Nf):
    return Nd*x**2 + Ne*x + Nf
xdata = np.ravel(x)
ydata = np.ravel(y_DiffLogNucRate)
plt.scatter(xdata, ydata, label = 'data')
popt, pcov = curve_fit(func, xdata, ydata)
Nd, Ne, Nf = popt
x_line = np.arange(min(x), max(x), 0.001)
plt.plot(x_line, func(x_line, *popt), 'r-', label='fit: Nd=%5.3f, Ne=%5.3f, Nf=%5.3f' % tuple(popt))
plt.xlabel("log of T")
plt.ylabel("Difference between target and measured nucleation rate (logged)")
plt.legend()
plt.show()