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
T = dropna_raw_df["Isothermal Temp"]
T = T + 273.15

#Domain y-values
y_NucRate = dropna_raw_df["Mean CB Nucleation Rate (#/s)"]

#Creating domain knowledge plots - IndTime wrt SS
x = T/100
def func(x, Nc, Nd):
    return Nc*x + Nd
xdata = np.ravel(x)
ydata = np.ravel(y_NucRate)
plt.scatter(xdata, ydata, label = 'data')
popt, _ = curve_fit(func, xdata, ydata)
Nc, Nd = popt
x_line = np.arange(min(x), max(x), 0.001)
plt.plot(x_line, func(x_line, *popt), 'r-', label='fit: Nc=%5.3f, Nd=%5.3f' % tuple(popt))
plt.xlabel("T /K/100")
plt.ylabel("Nucleation rate /#s-1")
plt.legend()
plt.show()

#Set the target parameters
from main import Target_NR
NucRate_Target_cps = Target_NR

#Creating y-values for a wider range of x-data from the analytical model above
T = np.arange(0, 6, 0.01)
y_SIM = Nc*T + Nd
y_SIM_Diff = abs(NucRate_Target_cps - y_SIM)
plt.plot(T, y_SIM_Diff, label = "Analytical model of domain knowledge (mathematically transformed)")
'''def func(x, Nd, Ne, Nf):
    return Nd*x**2 + Ne*x + Nf
popt, _ = curve_fit(func, T, y_SIM_Diff)
Nd, Ne, Nf = popt
plt.plot(T, func(T, *popt), 'r-', label='fit: Nd=%5.3f, Ne=%5.3f, Nf=%5.3f' % tuple(popt))'''
plt.xlabel("Temperature /K/100")
plt.ylabel("Difference between target nucleation rate and measured nucleation rate /#s-1")
plt.legend()
plt.show()


