import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#Set the target parameters
from main import Target_IT
IndTime_Target_s = Target_IT
IndTime_Target_s_log = math.log10(IndTime_Target_s)

#Load the raw data file
#raw_df = pd.read_excel(r'H:\Documents\CHEM ENG\MACHINE LEARNING\Bayesian Optimisation\EtOH_LAMV_DoE.xlsx')
from main import raw_df
raw_df_sel = raw_df[["SS", "Isothermal Temp", "Median CB Induction Time (s)"]]
dropna_raw_df = raw_df_sel.dropna()

#Process conditions (x-variables)
SS = dropna_raw_df["SS"]
SS = 10*np.log10(SS)
T = dropna_raw_df["Isothermal Temp"]
#T = (1/(T+273.15))
T = np.log10(T)

#y-values
y_IndTime = dropna_raw_df["Median CB Induction Time (s)"]
y_IndTime_log = np.log10(y_IndTime)
y_DiffLogIndTime = abs(IndTime_Target_s_log - y_IndTime_log)

#Creating plots - IndTime wrt SS
x = SS
def func(x, Ia, Ib, Ic):
    return Ia*x**2 + Ib*x + Ic
xdata = np.ravel(x)
ydata = np.ravel(y_DiffLogIndTime)
plt.scatter(xdata, ydata, label = 'data')
popt, _ = curve_fit(func, xdata, ydata)
Ia, Ib, Ic = popt
x_line = np.arange(min(x), max(x), 0.001)
plt.plot(x_line, func(x_line, *popt), 'r-', label='fit: Ia=%5.3f, Ib=%5.3f, Ic=%5.3f' % tuple(popt))
plt.xlabel("log of SS")
plt.ylabel("Difference between target and measured induction time (logged)")
plt.legend()
plt.show()

'''#Creating plots - IndTime wrt T
x = T
def func(x, Id, Ie, If):
    return Id*x**2 + Ie*x + If
xdata = np.ravel(x)
ydata = np.ravel(y_DiffLogIndTime)
plt.scatter(xdata, ydata, label = 'data')
popt, pcov = curve_fit(func, xdata, ydata)
Id, Ie, If = popt
x_line = np.arange(min(x), max(x), 0.001)
plt.plot(x_line, func(x_line, *popt), 'r-', label='fit: Id=%5.3f, Ie=%5.3f, If=%5.3f' % tuple(popt))
plt.xlabel("log of T")
plt.ylabel("Difference between target and measured induction time (logged)")
plt.legend()
plt.show()'''