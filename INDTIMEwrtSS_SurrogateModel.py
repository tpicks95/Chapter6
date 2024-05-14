import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#Load the raw data file
#raw_df = pd.read_excel(r'H:\Documents\CHEM ENG\MACHINE LEARNING\Bayesian Optimisation\EtOH_LAMV_DoE.xlsx')
from main import raw_df
raw_df_sel = raw_df[["SS", "Isothermal Temp", "Median CB Induction Time (s)"]]
dropna_raw_df = raw_df_sel.dropna()

#Process conditions (x-variables)
SS = dropna_raw_df["SS"]

#Domain y-values
y_IndTime = dropna_raw_df["Median CB Induction Time (s)"]

#Creating domain knowledge plots - IndTime wrt SS
x = SS
def func(x, Ia, Ib):
    return Ia*np.exp(-Ib*x)
xdata = np.ravel(x)
ydata = np.ravel(y_IndTime)
plt.scatter(xdata, ydata, label = 'data')
popt, _ = curve_fit(func, xdata, ydata)
Ia, Ib = popt
x_line = np.arange(min(x), max(x), 0.001)
plt.plot(x_line, func(x_line, *popt), 'r-', label='fit: Ia=%5.3f, Ib=%5.3f' % tuple(popt))
plt.xlabel("SS")
plt.ylabel("Induction time /s")
plt.legend()
plt.show()

#Set the target parameters
from main import Target_IT
IndTime_Target_s = Target_IT
IndTime_Target_s_log = np.log(IndTime_Target_s)

#Creating y-values for a wider range of x-data from the analytical model above
SS = np.arange(0, 5, 0.01)
y_SIM = Ia*np.exp(-Ib*SS)
y_SIM_Diff = abs(np.log(IndTime_Target_s) - np.log(y_SIM))
plt.plot(SS, y_SIM_Diff, label = "Analytical model of domain knowledge (mathematically transformed)")
'''def func(x, Ia, Ib, Ic):
    return Ia*x**2 + Ib*x + Ic
popt, _ = curve_fit(func, SS, y_SIM_Diff)
Ia, Ib, Ic = popt
plt.plot(SS, func(SS, *popt), 'r-', label='fit: Ia=%5.3f, Ib=%5.3f, Ic=%5.3f' % tuple(popt))'''
plt.xlabel("SS")
plt.ylabel("Difference between target tind and measured tind /s")
plt.legend()
plt.show()


