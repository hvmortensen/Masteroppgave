import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'cm'
mpl.rcParams['font.sans-serif'] = 'Times New Roman'
mpl.rcParams['font.family'] = 'Times New Roman'


D1 = np.array([0.0, 0.1, 0.2, 0.3, 0.5, 1.0])
dl1 = len(D1)

# T47D/P-filer
h0 = np.loadtxt("H3-10_data.txt")
h1 = np.loadtxt("H3-11_henrik.txt")
h2 = np.loadtxt("H3-11_nina.txt")
h3 = np.loadtxt("H3-11_ingunn.txt")
h4 = np.loadtxt("H3-11_jenny.txt")
h5 = np.loadtxt("H3-23_data.txt")
h6 = np.loadtxt("H3-23_hist_data.txt")

H = [h0, h1, h2, h3, h4, h5, h6]
no1 = len(H)

h = np.zeros((no1,dl1))
p = np.zeros((no1,dl1))
hs = np.zeros((no1,dl1))
ps = np.zeros((no1,dl1))
for i in range(dl1):
    for j in range(no1):
        h[j,i] = np.mean(H[j][i])/np.mean(H[j][0])
        p[j,i] = np.mean(H[j][i+6])/np.mean(H[j][6])
        hs[j,i] = np.std(H[j][i])/np.mean(H[j][0])
        ps[j,i] = np.std(H[j][i+6])/np.mean(H[j][6])

# print(h)

D2 = np.array([0.0, 0.1, 0.2, 0.3, 0.5])
dl2 = len(D2)

# T47D/P/T-filer
hh0 = np.loadtxt("H3-18_henrik.txt")
hh1 = np.loadtxt("H3-18_nina.txt")
hh2 = np.loadtxt("H3-18_ingunn.txt")
hh3 = np.loadtxt("H3-18_jenny.txt")
hh4 = np.loadtxt("H3-20_data1.txt")
hh5 = np.loadtxt("H3-25_data.txt")
hh6 = np.loadtxt("H3-26_data.txt")
hh7 = np.loadtxt("H3-27_data.txt") # 0.3T udtværet plot, datapunkter kan ikke regnes med

HH = [hh4, hh5, hh6, hh7]
no2 = len(HH)

hh = np.zeros((no2,dl2))
pp = np.zeros((no2,dl2))
tt = np.zeros((no2,dl2))
hhs = np.zeros((no2,dl2))
pps = np.zeros((no2,dl2))
tts = np.zeros((no2,dl2))
for i in range(dl2):
    for j in range(no2):
        hh[j,i] = np.mean(HH[j][i])/np.mean(HH[j][0])
        pp[j,i] = np.mean(HH[j][i+5])/np.mean(HH[j][5])
        tt[j,i] = np.mean(HH[j][i+10])/np.mean(HH[j][10])
        hhs[j,i] = np.std(HH[j][i])/np.mean(HH[j][0])
        pps[j,i] = np.std(HH[j][i+5])/np.mean(HH[j][5])
        tts[j,i] = np.std(HH[j][i+10])/np.mean(HH[j][10])

ha = np.zeros((dl1, no1+no2))
pa = np.zeros((dl1, no1+no2))
ta = np.zeros((dl2, no2))
for i in range(dl1):
    for j in range(no1):
        ha[i,j] = h[j,i]
        pa[i,j] = p[j,i]
for i in range(dl2):
    for j in range(no2):
        ha[i,j+no1] = hh[j,i]
        pa[i,j+no1] = pp[j,i]
        ta[i,j] = tt[j,i]


HM = np.zeros(dl2)
PM = np.zeros(dl2)
TM = np.zeros(dl2)
HS = np.zeros(dl2)
PS = np.zeros(dl2)
TS = np.zeros(dl2)
for i in range(dl2):
    HM[i] = np.mean(ha[i])
    PM[i] = np.mean(pa[i])
    TM[i] = np.mean(ta[i])
    HS[i] = np.std(ha[i])/np.sqrt(no1+no2)
    PS[i] = np.std(pa[i])/np.sqrt(no1+no2)
    TS[i] = np.std(ta[i])/np.sqrt(no1+no2)


FS = 17
plt.errorbar(D2,HM,yerr=HS,uplims=True,lolims=True,label="T47D")
plt.errorbar(D2,PM,yerr=HS,uplims=True,lolims=True,label="T47D-P")
plt.errorbar(D2,TM,yerr=HS,uplims=True,lolims=True,label="T47D-T")
plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
plt.xlabel("Dosis (Gy)",fontsize=FS)
plt.ylabel("Mitotisk indeks (%)",fontsize=FS)
plt.tick_params(axis='both', which='major', labelsize=FS)
plt.legend(fontsize=FS-2)

plt.tight_layout()
plt.savefig("H3_allbydose.pdf")
plt.show()
