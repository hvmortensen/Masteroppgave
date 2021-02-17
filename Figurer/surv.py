import numpy as np
import matplotlib.pyplot as plt


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

FS = 14 # fontsixe til superoverskrift
fs = FS - 2  # fontsize til lengend()
FFS = FS + 1

# plt.rcParams.update({'font.size': 12})
ax = plt.subplot(1, 1, 1)
plt.semilogy(D,S(D), linewidth=2, label="Overlevelsesfraktion")
plt.semilogy(D,Sn(D), "r-.", linewidth=2, label="LQ-model")
plt.semilogy(xr,line_r(xr),"--", label="$\\alpha_r$")
plt.semilogy(xs,line_s(xs),"--", label="$\\alpha_s$")
# plt.semilogy(Dc, np.exp(-a*Dc - b*Dc**2))
# plt.semilogy(Dc,"o")
plt.plot(Dc, S(Dc), "ko", label="$D_c$")
ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
ax.axis([0, 2, 0.45, 1.09])
ax.tick_params(axis='both', which='major', labelsize=fs)
plt.yticks( [ 1.0, 0.9, 0.8, 0.7, 0.6, 0.5] )
plt.xticks( [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0] )
ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
# # xticks(size=f)
plt.ylabel('Overlevelsesfraktion',fontsize=fs)
plt.xlabel('Dose (Gy)',fontsize=fs)
plt.legend(fontsize=fs)
# minorticks_on()

# locmin = LogLocator(subs=(0.51,0.52,0.53,0.54,0.55,0.56,0.57,0.58,0.59,0.61,
# 0.62,0.63,0.64,0.65,0.66,0.67,0.68,0.69,0.71,0.72,0.73,0.74,0.75,0.76,0.77,
# 0.78,0.79,0.81,0.82,0.83,0.84,0.85,0.86,0.87,0.88,0.89,0.91,0.92,0.93,0.94,
# 0.95,0.96,0.97,0.98,0.99,1.01,1.02,1.03,1.04,1.05,1.06,1.07,1.08,1.09
#  ),numticks=12)
# ax.yaxis.set_minor_locator(locmin)
# ax.yaxis.set_minor_formatter(NullFormatter())

plt.tight_layout()
plt.savefig("Survivalcurve.pdf")
plt.show()
