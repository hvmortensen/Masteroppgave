import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'cm'
mpl.rcParams['font.sans-serif'] = 'Times New Roman'
mpl.rcParams['font.family'] = 'Times New Roman'

D = np.linspace(0,10,1501)

a_r  = 0.23
a_s = 1.75
b   = 0.1# 0.022
Dc  = 0.3

def a(x):
    return a_r*(1 + (a_s/a_r - 1)*np.exp(-x/Dc))

def S(x):
    return np.exp(-a(x)*x - b*x**2)

def Sn(x):
    return np.exp(-a_r*x - b*x**2)

xr = np.linspace(0,0.7)
xs = np.linspace(0,0.25)

def line_r(x):
    return np.exp(-a_r*x)

def line_s(x):
    return np.exp(-a_s*x)

FS = 14+3 # fontsixe til superoverskrift
fs = FS + 1  # fontsize til legend()

ax = plt.subplot(1, 1, 1)
plt.semilogy(D,S(D), linewidth=2, label="Overlevelsesfraktion")
plt.semilogy(D,Sn(D), "r-.", linewidth=2, label="LQ-model")
plt.semilogy(xr,line_r(xr),"--")
plt.semilogy(xs,line_s(xs),"--")
plt.plot(Dc, S(Dc), "ko")

ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
ax.axis([0, 2, 0.45, 1.09])
ax.tick_params(axis='both', which='major', labelsize=fs)
plt.yticks( [ 1.0, 0.9, 0.8, 0.7, 0.6, 0.5] )
plt.xticks( [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0] )
ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
plt.ylabel('Overlevelsesfraktion',fontsize=fs)
plt.xlabel('Dosis (Gy)',fontsize=fs)
plt.legend(fontsize=fs)
ax.text(0.26,0.65 ,"$\\alpha_s$",fontsize=fs)
ax.text(0.67,0.87 ,"$\\alpha_r$",fontsize=fs)
ax.text(0.28,0.73 ,"$D_c$",fontsize=fs)
plt.tight_layout()
plt.savefig("Survivalcurve.pdf")
plt.show()
