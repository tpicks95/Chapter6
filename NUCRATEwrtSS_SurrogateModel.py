import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#Load the raw data file
#raw_df = pd.read_excel(r'H:\Documents\CHEM ENG\MACHINE LEARNING\Bayesian Optimisation\EtOH_LAMV_DoE.xlsx')
from main import raw_df
raw_df_sel = raw_df[["SS", "Isothermal Temp", "Mean CB Nucleation Rate (#/s)"]]
dropna_raw_df = raw_df_sel.dropna()

#Process conditions (x-variables)
SS = dropna_raw_df["SS"]

#Domain y-values
y_NucRate = dropna_raw_df["Mean CB Nucleation Rate (#/s)"]

#Creating domain knowledge plots - IndTime wrt SS
x = SS
def func(x, Na, Nb):
    return Na*np.exp(Nb*x)
xdata = np.ravel(x)
ydata = np.ravel(y_NucRate)
plt.scatter(xdata, ydata, label = 'data')
popt, _ = curve_fit(func, xdata, ydata)
Na, Nb = popt
x_line = np.arange(min(x), max(x), 0.001)
plt.plot(x_line, func(x_line, *popt), 'r-', label='fit: Na=%5.3f, Nb=%5.3f' % tuple(popt))
plt.xlabel("SS")
plt.ylabel("Nucleation rate /#s-1")
plt.legend()
plt.show()

#Set the target parameters
from main import Target_NR
NucRate_Target_cps = Target_NR

#Creating y-values for a wider range of x-data from the analytical model above
SS = np.arange(0, 5, 0.01)
y_SIM = Na*np.exp(Nb*SS)
y_SIM_Diff = abs(np.log(NucRate_Target_cps) - np.log(y_SIM))
plt.plot(SS, y_SIM_Diff, label = "Analytical model of domain knowledge (mathematically transformed)")
'''def func(x, Na, Nb, Nc):
    return Na*x**2 + Nb*x + Nc
popt, _ = curve_fit(func, SS, y_SIM_Diff)
Na, Nb, Nc = popt
plt.plot(SS, func(SS, *popt), 'r-', label='fit: Na=%5.3f, Nb=%5.3f, Nc=%5.3f' % tuple(popt))'''
plt.xlabel("SS")
plt.ylabel("Difference between target nucleation rate and measured nucleation rate /#s-1")
plt.legend()
plt.show()


