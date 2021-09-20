import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'cm'
mpl.rcParams['font.sans-serif'] = 'Times New Roman'
mpl.rcParams['font.family'] = 'Times New Roman'

#antal forsøg
N = 3

D = np.array([0, 0.1, 0.2, 0.3, 0.5, 1.0])
d19h = np.array([0, 0.1, 0.3, 0.5, 1.0])
d19p = np.array([0, 0.1, 0.2, 0.3, 1.0])


h100 = np.array([2.8, 3.0, 2.7, 2.6, 2.6])
h101 = np.array([2.8, 2.7, 2.6, 2.6, 2.5])
h102 = np.array([2.5, 2.4, 2.4, 2.4, 2.2])
h103 = np.array([2.8, 2.7, 2.6, 2.6, 2.4])
h105 = np.array([1.9, 2.1, 1.9, 1.8, 1.6])
h10x = np.array([1.2, 1.4, 1.1, 1.0, 0.9])
p100 = np.array([3.8, 3.7, 3.1, 3.6, 3.3])
p101 = np.array([2.4, 2.5, 2.7, 2.6, 2.8])
p102 = np.array([2.5, 2.2, 2.3, 2.3, 2.3])
p103 = np.array([2.5, 2.6, 2.3, 2.4, 2.5])
p105 = np.array([1.4, 1.3, 1.1, 1.3, 1.5])
p10x = np.array([2.0, 1.9, 1.9, 1.8, 1.4])


h110 = np.array([2.5, 2.4, 2.0, 2.0, 2.1])
h111 = np.array([2.4, 2.6, 2.1, 2.2, 2.3])
h112 = np.array([2.7, 2.5, 2.4, 2.4, 2.1])
h113 = np.array([1.9, 1.7, 1.5, 1.2, 1.3])
h115 = np.array([1.6, 1.6, 1.4, 1.1, 1.3])
h11x = np.array([2.0, 1.5, 2.0, 1.4, 1.2])
p110 = np.array([2.8, 2.8, 2.4, 2.2, 2.5])
p111 = np.array([1.9, 2.0, 2.0, 1.3, 1.8])
p112 = np.array([2.6, 1.7, 1.7, 1.6, 2.3])
p113 = np.array([1.3, 1.7, 1.3, 1.2, 1.2])
p115 = np.array([1.3, 1.7, 1.3, 0.9, 1.1])
p11x = np.array([0.9, 1.0, 0.7, 0.5, 1.1])


h190a = np.array([3.9, 4.1, 4.7, 3.7, 4.0])
h190b = np.array([3.0, 3.2, 3.1, 3.8, 3.6])
h191 = np.array([3.7, 3.8, 3.8, 3.6, 4.2])
h193 = np.array([3.2, 3.7, 3.3, 3.0, 3.0])
h195 = np.array([2.7, 3.2, 2.6, 2.9, 2.8])
h19x = np.array([3.2, 2.8, 2.6, 2.9, 2.8])
p190a = np.array([3.5, 3.7, 3.2, 3.3, 3.9])
p190b = np.array([3.6, 3.7, 3.4, 3.8, 4.1])
p191 = np.array([3.4, 4.5, 3.1, 3.3, 3.5])
p192 = np.array([3.8, 4.8, 3.1, 3.7, 3.8])
p193 = np.array([1.7, 2.0, 2.0, 2.7, 2.9])
p19x = np.array([2.1, 2.8, 2.0, 2.0, 2.2])


### Skaleringsfaktor til mitotisk indeks
h10k = np.mean(h100)/100
p10k = np.mean(p100)/100

h11k = np.mean(h110)/100
p11k = np.mean(p110)/100

h19k = np.mean(np.array([h190a,h190b]))/100
p19k = np.mean(np.array([p190a,p190b]))/100

### Mitotisk indeks og standardafvig i gating
### Standardafvig er skaleret med samme konstant som mitotisk indeks
### dvs. gennemsnit af kontrolprøvens mitotiske indeks
h100m = np.mean(h100)/h10k
h101m = np.mean(h101)/h10k
h102m = np.mean(h102)/h10k
h103m = np.mean(h103)/h10k
h105m = np.mean(h105)/h10k
h10xm = np.mean(h10x)/h10k
p100m = np.mean(p100)/p10k
p101m = np.mean(p101)/p10k
p102m = np.mean(p102)/p10k
p103m = np.mean(p103)/p10k
p105m = np.mean(p105)/p10k
p10xm = np.mean(h10x)/p10k

h100s = np.std(h100)/h10k
h101s = np.std(h101)/h10k
h102s = np.std(h102)/h10k
h103s = np.std(h103)/h10k
h105s = np.std(h105)/h10k
h10xs = np.std(h10x)/h10k
p100s = np.std(p100)/p10k
p101s = np.std(p101)/p10k
p102s = np.std(p102)/p10k
p103s = np.std(p103)/p10k
p105s = np.std(p105)/p10k
p10xs = np.std(h10x)/p10k


h110m = np.mean(h110)/h11k
h111m = np.mean(h111)/h11k
h112m = np.mean(h112)/h11k
h113m = np.mean(h113)/h11k
h115m = np.mean(h115)/h11k
h11xm = np.mean(h11x)/h11k
p110m = np.mean(p110)/p11k
p111m = np.mean(p111)/p11k
p112m = np.mean(p112)/p11k
p113m = np.mean(p113)/p11k
p115m = np.mean(p115)/p11k
p11xm = np.mean(h11x)/p11k

h110s = np.std(h110)/h11k
h111s = np.std(h111)/h11k
h112s = np.std(h112)/h11k
h113s = np.std(h113)/h11k
h115s = np.std(h115)/h11k
h11xs = np.std(h11x)/h11k
p110s = np.std(p110)/p11k
p111s = np.std(p111)/p11k
p112s = np.std(p112)/p11k
p113s = np.std(p113)/p11k
p115s = np.std(p115)/p11k
p11xs = np.std(h11x)/p11k

h190m = np.mean(np.array([h190a,h190b]))/h19k
h191m = np.mean(h191)/h19k
h193m = np.mean(h193)/h19k
h195m = np.mean(h195)/h19k
h19xm = np.mean(h19x)/h19k
p190m = np.mean(np.array([p190a,p190b]))/p19k
p191m = np.mean(p191)/p19k
p192m = np.mean(p192)/p19k
p193m = np.mean(p193)/p19k
p19xm = np.mean(h19x)/p19k

h190s = np.std(np.array([h190a,h190b]))/h19k
h191s = np.std(h191)/h19k
h193s = np.std(h193)/h19k
h195s = np.std(h195)/h19k
h19xs = np.std(h19x)/h19k
p190s = np.std(np.array([p190a,p190b]))/p19k
p191s = np.std(p191)/p19k
p192s = np.std(p192)/p19k
p193s = np.std(p193)/p19k
p19xs = np.std(h19x)/p19k



h10m = np.array([h100m, h101m, h102m, h103m, h105m, h10xm])
p10m = np.array([p100m, p101m, p102m, p103m, p105m, p10xm])
h10s = np.array([h100s, h101s, h102s, h103s, h105s, h10xs])
p10s = np.array([p100s, p101s, p102s, p103s, p105s, p10xs])

h11m = np.array([h110m, h111m, h112m, h113m, h115m, h11xm])
p11m = np.array([p110m, p111m, p112m, p113m, p115m, p11xm])
h11s = np.array([h110s, h111s, h112s, h113s, h115s, h11xs])
p11s = np.array([p110s, p111s, p112s, p113s, p115s, p11xs])

h19m = np.array([h190m, h191m,        h193m, h195m, h19xm])
p19m = np.array([p190m, p191m, p192m, p193m,        p19xm])
h19s = np.array([h190s, h191s,        h193s, h195s, h19xs])
p19s = np.array([p190s, p191s, p192s, p193s,        p19xs])

h0 = np.array([h100m, h110m, h190m])
h1 = np.array([h101m, h111m, h191m])
h2 = np.array([h102m, h112m])
h3 = np.array([h103m, h113m, h193m])
h5 = np.array([h105m, h115m, h195m])
hx = np.array([h10xm, h11xm, h19xm])
p0 = np.array([p100m, h110m, p190m])
p1 = np.array([p101m, h111m, p191m])
p2 = np.array([p102m, h112m, p192m])
p3 = np.array([p103m, h113m, p193m])
p5 = np.array([p105m, h115m])
px = np.array([p10xm, h11xm, p19xm])

h0m = np.mean(h0)
h1m = np.mean(h1)
h2m = np.mean(h2)
h3m = np.mean(h3)
h5m = np.mean(h5)
hxm = np.mean(hx)
p0m = np.mean(p0)
p1m = np.mean(p1)
p2m = np.mean(p2)
p3m = np.mean(p3)
p5m = np.mean(p5)
pxm = np.mean(px)

h0s = np.std(h0)/np.sqrt(len(h0))
h1s = np.std(h1)/np.sqrt(len(h1))
h2s = np.std(h2)/np.sqrt(len(h2))
h3s = np.std(h3)/np.sqrt(len(h3))
h5s = np.std(h5)/np.sqrt(len(h5))
hxs = np.std(hx)/np.sqrt(len(hx))
p0s = np.std(p0)/np.sqrt(len(p0))
p1s = np.std(p1)/np.sqrt(len(p1))
p2s = np.std(p2)/np.sqrt(len(p2))
p3s = np.std(p3)/np.sqrt(len(p3))
p5s = np.std(p5)/np.sqrt(len(p5))
pxs = np.std(px)/np.sqrt(len(px))

havg = np.array([h0m, h1m, h2m, h3m, h5m, hxm])
pavg = np.array([p0m, p1m, p2m, p3m, p5m, pxm])
hstd = np.array([h0s, h1s, h2s, h3s, h5s, hxs])
pstd = np.array([p0s, p1s, p2s, p3s, p5s, pxs])




filename = "flowcytdata.txt"
data = np.loadtxt(filename)
### alle førstemålinger norm. på gnsn af egen kohorte
# a er alle førstemålinger
M = data
a = M[0:,0]#/np.mean(M[0:,0])
b = M[0:,1]
c = M[0:,2]
d = M[0:,3]
e = M[0:,4]




# x er målingerne i første plot
x = M[0,0:]

# an = np.zeros_like(a)
# print(an)
FS = 17

norm = np.zeros_like(M)
NormArray = np.zeros((M.shape[1], M.shape[0]))
print("M.shape[0] = ", M.shape[0])
print("M.shape[1] = ", M.shape[1])
for i in range(M.shape[0]):
    norm[i] = M[i]/np.mean(M[i])
    print(norm[i])
    for j in range(M.shape[1]):
        NormArray[j,i] = norm[i,j]

figa, axa = plt.subplots(1,M.shape[1], figsize=(14, 6), sharey="all")
for i in range(M.shape[1]):
    avg = np.mean(NormArray[i])
    sem = np.std(NormArray[i])/np.sqrt(len(NormArray[i]))
    axa[i].axhline(y=avg,label="gn: %.3f±%.3f"%(avg,sem), color="red")
    axa[i].plot(NormArray[i], ".", label="norm. måling")
    axa[i].legend(loc=9, fontsize=FS-2)
plt.tight_layout()
plt.show()

# plt.axhline(y=np.mean(a/np.mean(x)))
# plt.show()



fig, ax = plt.subplots(2,3, figsize=(14,8.5),sharey="all")
ax[0,0].errorbar(D, h10m, yerr=h10s,  label="T47D")
ax[0,0].errorbar(D, p10m, yerr=p10s,  label="T47D-P")
ax[0,0].text(0.05, 0.95, "H3-10 ", transform=ax[0,0].transAxes)
ax[0,1].errorbar(D, h11m, yerr=h11s,  label="T47D")
ax[0,1].errorbar(D, p11m, yerr=p11s,  label="T47D-P")
ax[0,1].text(0.05, 0.95, "H3-11 ", transform=ax[0,1].transAxes)
ax[0,2].errorbar(d19h, h19m, yerr=h19s,  label="T47D")
ax[0,2].errorbar(d19p, p19m, yerr=p19s,  label="T47D-P")
ax[0,2].text(0.05, 0.95, "H3-11 ", transform=ax[0,2].transAxes)


for i in range(2):
    for j in range(3):
        ax[i,j].legend(fontsize=FS-2)
        ax[i,j].set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax[i,j].set_xlabel("Dosis (Gy)",fontsize=FS)
        ax[i,j].tick_params(axis='both', which='major', labelsize=FS)
        ax[i,0].set_ylabel("Mitotisk index (%)",fontsize=FS)
plt.tight_layout()
plt.savefig("mitoticindex.pdf")
# plt.show()

plt.errorbar(D, havg, yerr=hstd, uplims=True, lolims=True, label="T47D")
plt.errorbar(D, pavg, yerr=pstd, uplims=True, lolims=True, label="T47D-P")
plt.xlabel("Dosis (Gy)",fontsize=FS)
plt.ylabel("Mitotisk indeks (%)",fontsize=FS)
plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
plt.tick_params(axis='both', which='major', labelsize=FS)
plt.legend(fontsize=FS-2)
plt.tight_layout()
plt.savefig("mitoticindex_all.pdf")
# plt.show()
