import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'cm'
mpl.rcParams['font.sans-serif'] = 'Times New Roman'
mpl.rcParams['font.family'] = 'Times New Roman'

D = np.array([0.0, 0.1, 0.2, 0.3, 0.5, 1.0])

a = np.loadtxt("T47D01.txt")
b = np.loadtxt("T47D02.txt")
c = np.loadtxt("T47D03.txt")
d = np.loadtxt("T47D05.txt")
e = np.loadtxt("T47D10.txt")

pa = np.loadtxt("T47D-P01.txt")
pb = np.loadtxt("T47D-P02.txt")
pc = np.loadtxt("T47D-P03.txt")
pd = np.loadtxt("T47D-P05.txt")
pe = np.loadtxt("T47D-P10.txt")

ta = np.loadtxt("T47D-T01.txt")
tb = np.loadtxt("T47D-T02.txt")
tc = np.loadtxt("T47D-T03.txt")
td = np.loadtxt("T47D-T05.txt")
te = np.loadtxt("T47D-T10.txt")

h = [a,b,c,d,e]
p = [pa,pb,pc,pd,pe]
t = [ta,tb,tc,td,te]

h00 = np.ones(int(a.shape[0]/2))
h01 = np.zeros(int(a.shape[0]/2))
h02 = np.zeros(int(b.shape[0]/2))
h03 = np.zeros(int(c.shape[0]/2))
h05 = np.zeros(int(d.shape[0]/2))
h10 = np.zeros(int(e.shape[0]/2))

p00 = np.ones(int(pa.shape[0]/2))
p01 = np.zeros(int(pa.shape[0]/2))
p02 = np.zeros(int(pb.shape[0]/2))
p03 = np.zeros(int(pc.shape[0]/2))
p05 = np.zeros(int(pd.shape[0]/2))
p10 = np.zeros(int(pe.shape[0]/2))

t00 = np.ones(int(ta.shape[0]/2))
t01 = np.zeros(int(ta.shape[0]/2))
t02 = np.zeros(int(tb.shape[0]/2))
t03 = np.zeros(int(tc.shape[0]/2))
t05 = np.zeros(int(td.shape[0]/2))
t10 = np.zeros(int(te.shape[0]/2))

for i in range(int(a.shape[0]/2)):
    h01[i] = np.mean(a[i*2+1])/np.mean(a[i*2])
for i in range(int(b.shape[0]/2)):
    h02[i] = np.mean(b[i*2+1])/np.mean(b[i*2])
for i in range(int(c.shape[0]/2)):
    h03[i] = np.mean(c[i*2+1])/np.mean(c[i*2])
for i in range(int(d.shape[0]/2)):
    h05[i] = np.mean(d[i*2+1])/np.mean(d[i*2])
for i in range(int(e.shape[0]/2)):
    h10[i] = np.mean(e[i*2+1])/np.mean(e[i*2])

for i in range(int(pa.shape[0]/2)):
    p01[i] = np.mean(pa[i*2+1])/np.mean(pa[i*2])
for i in range(int(pb.shape[0]/2)):
    p02[i] = np.mean(pb[i*2+1])/np.mean(pb[i*2])
for i in range(int(pc.shape[0]/2)):
    p03[i] = np.mean(pc[i*2+1])/np.mean(pc[i*2])
for i in range(int(pd.shape[0]/2)):
    p05[i] = np.mean(pd[i*2+1])/np.mean(pd[i*2])
for i in range(int(pe.shape[0]/2)):
    p10[i] = np.mean(pe[i*2+1])/np.mean(pe[i*2])

for i in range(int(ta.shape[0]/2)):
    t01[i] = np.mean(ta[i*2+1])/np.mean(ta[i*2])
for i in range(int(tb.shape[0]/2)):
    t02[i] = np.mean(tb[i*2+1])/np.mean(tb[i*2])
for i in range(int(tc.shape[0]/2)):
    t03[i] = np.mean(tc[i*2+1])/np.mean(tc[i*2])
for i in range(int(td.shape[0]/2)):
    t05[i] = np.mean(td[i*2+1])/np.mean(td[i*2])
for i in range(int(te.shape[0]/2)):
    t10[i] = np.mean(te[i*2+1])/np.mean(te[i*2])


H = np.array([h00, h01, h02, h03, h05, h10])
P = np.array([p00, p01, p02, p03, p05, p10])
T = np.array([t00, t01, t02, t03, t05, t10])

HM = np.zeros(len(H))
PM = np.zeros(len(P))
TM = np.zeros(len(T))
HS = np.zeros(len(H))
PS = np.zeros(len(P))
TS = np.zeros(len(T))
for i in range(len(H)):
    HM[i] = np.mean(H[i])
    HS[i] = np.std(H[i])/np.sqrt(len(H[i]))
for i in range(len(P)):
    PM[i] = np.mean(P[i])
    PS[i] = np.std(P[i])/np.sqrt(len(P[i]))
for i in range(len(T)):
    TM[i] = np.mean(T[i])
    TS[i] = np.std(T[i])/np.sqrt(len(T[i]))

FS = 17 + 2

plt.errorbar(D,HM*100, yerr=HS*100, uplims=True, lolims=True,label="T47D")
plt.errorbar(D,PM*100, yerr=PS*100, uplims=True, lolims=True,label="T47D-P")
plt.errorbar(D,TM*100, yerr=TS*100, uplims=True, lolims=True,label="T47D-T")
plt.legend(fontsize=FS)
plt.xticks(fontsize=FS)
plt.yticks(fontsize=FS)
plt.xlabel("Dosis (Gy)", fontsize=FS)
plt.ylabel("Mitotisk indeks (%)", fontsize=FS)
plt.tight_layout()
plt.savefig("H3-allesammen.pdf")
plt.show()
