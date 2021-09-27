import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'cm'
mpl.rcParams['font.sans-serif'] = 'Times New Roman'
mpl.rcParams['font.family'] = 'Times New Roman'


SSD40data = np.loadtxt("SSD40_3s-13s_60secbreak.txt")
M = SSD40data

t = np.array([3,4,5,6,7,8,9,10,11,12,13])
k = len(t)              # Antal målingskohorter i hvert datasæt
n = int(M.shape[0]/k)   # Antallet af datasæt
print(n)

### Normalisér førstemålinger på gennemsnit af kohorten
NormArray = np.zeros((k,n))
for i in range(n):      # hele filen
    for j in range(k):  # hver linje i hele filen
        NormArray[j,i] = (M[j + i*k,0]/np.mean(M[j + k*i]))

### Lav arrays for hver eksponeringstid 3s - 13s
norms = np.zeros(k)
stds = np.zeros(k)
for i in range(k):
    norms[i] = np.mean(NormArray[i])
    stds[i] = np.std(NormArray[i])/np.sqrt(n)

norms = (norms-1)*100
stds = stds*100
T = np.linspace(3,13, 1001)
tt = np.array([0,1,2,3,4,5,6,7,8,9,10])
x_data = t
y_data = norms
log_x_data = np.log(x_data)
log_y_data = np.log(y_data)

curve_fit = np.polyfit(x_data, log_y_data, 1)
# print(curve_fit)

xx = 850
ttt = 0.77
print(norms)
print(xx*np.exp(-ttt*3))
print(xx*np.exp(-ttt*4))
print(xx*np.exp(-ttt*5))
print(xx*np.exp(-ttt*6))
print(xx*np.exp(-ttt*7))
print(xx*np.exp(-ttt*8))
print(xx*np.exp(-ttt*9))
print(xx*np.exp(-ttt*10))
print(xx*np.exp(-ttt*11))
print(xx*np.exp(-ttt*12))
print(xx*np.exp(-ttt*13))
FS = 14+3
plt.plot(t, norms, "r*")
plt.plot(T, 850*np.exp(-0.77*T), "g", label="$f(t)=850e^{-0.77t}$")
plt.fill_between(t, norms-stds, norms+stds, alpha=0.3,label="SEM")
plt.axhline(y=0, linestyle='dotted', color="k")
plt.ylabel("Første målings afvig fra gennemsnittet (%)",fontsize=FS)
plt.xlabel("Eksponeringstid (s)",fontsize=FS)
plt.xticks([3,4,5,6,7,8,9,10,11,12,13])
plt.tick_params(axis='both', which='major', labelsize=FS)
plt.legend(fontsize=FS-2)
# plt.legend(["%.0fs, %.1f%%"%(t[0],norms[0])],fontsize=FS)
plt.tight_layout()
plt.savefig("calibration_constant.pdf")
plt.show()

print(norms)
