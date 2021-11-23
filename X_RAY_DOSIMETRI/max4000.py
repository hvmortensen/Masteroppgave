import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cycler
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox, AnchoredOffsetbox
import matplotlib.image as mpimg
import matplotlib as mpl

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'cm'
mpl.rcParams['font.sans-serif'] = 'Times New Roman'
mpl.rcParams['font.family'] = 'Times New Roman'

cn = [0,1,2,3,4,5,6,7,8,9,10, 11,12, 20, 21, 22, 23, 24]
cf = [0,1,2,3,4,5,6,7,8,9,10,11,12,18,19]
ck = [0,1,2,3,4,5,6,10,11]

m = 60
tn = [0, 28, m+21, 2*m+10, 3*m, 3*m+47, 4*m+38, 5*m+27, 6*m+10, 6*m+55, 7*m+38, 8*m+21, 9*m+7, 13*m+46, 14*m+18, 14*m+50, 15*m+24, 15*m+58 ]
tf = [0, 20, 51, m+24, 1*m+55, 2*m+24, 2*m+54, 3*m+23, 3*m+51, 4*m+19, 4*m+48, 5*m+17, 5*m+46, 8*m+36, 9*m+6]
tk = [0, 15, 39, m+7, m+33, 2*m, 2*m+27, 4*m+19, 4*m+43]

T = np.linspace(0,15*m+58)

fl = np.polyfit(tn[:],cn[:],1)
a,b = fl
def f(x):
    return a*x + b
#
gl = np.polyfit(tf[:],cf[:],1)
c,d = gl
def g(x):
    return c*x + d
#
hl = np.polyfit(tk[:],ck[:],1)
m,l = hl
def h(x):
    return m*x + l
FS = 17+3;
FS = FS-3
fs = FS-2  # fontsize til legend()
plt.plot(tn,cn,"o")
plt.plot(tf,cf,"o")
plt.plot(tk,ck,"o")
plt.plot(T, f(T),"b", label="MAX4000 + kabel + FC65-G")
plt.plot(T, g(T),"r", label="MAX4000 + kabel")
plt.plot(T, h(T),"g", label="MAX4000")
plt.xlabel("Tid (min)",fontsize=FS)
plt.ylabel("Måling (nC)",fontsize=FS)
tx= [0,120,240,360,480,600, 720, 840,960]
xlabels = ["0", "2", "4", "6", "8", "10", "12", "14", "16"]
plt.xticks(tx, xlabels)
cx= [0,5,10,15,20,25,30,35]
ylabels = ["0.00", "0.05", "0.10", "0.15", "0.20", "0.25", "0.30","0.35"]
plt.yticks(cx, ylabels)
plt.tick_params(axis='both', which='major', labelsize=FS)
plt.legend(fontsize=FS-1)
plt.tight_layout()
plt.savefig("Max4000tidsfejl.pdf")
plt.show()
