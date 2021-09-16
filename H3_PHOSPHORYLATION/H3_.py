import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'cm'
mpl.rcParams['font.sans-serif'] = 'Times New Roman'
mpl.rcParams['font.family'] = 'Times New Roman'

D = np.array([0, 0.1, 0.2, 0.3, 0.5, 1.0])


h110a = np.array([2.8, 2.6, 2.7, 2.6])
h111a = np.array([2.6, 2.8, 3.2, 2.8])
h112a = np.array([2.7, 2.6, 2.9, 2.7])
h113a = np.array([1.8, 2.0, 1.8, 1.7])
h115a = np.array([1.8, 1.9, 1.9, 2.2])
h11xa = np.array([1.9, 2.5, 2.5, 2.5])
p110a = np.array([3.4, 3.2, 3.6, 2.9])
p111a = np.array([2.3, 2.0, 3.0, 2.0])
p112a = np.array([3.2, 2.8, 3.3, 2.7])
p113a = np.array([2.2, 2.2, 2.4, 2.2])
p115a = np.array([2.0, 2.0, 1.9, 1.8])
p11xa = np.array([1.1, 1.6, 1.2, 1.1])

h110am = np.mean(h110a)
h111am = np.mean(h111a)
h112am = np.mean(h112a)
h113am = np.mean(h113a)
h115am = np.mean(h115a)
h11xam = np.mean(h11xa)
p110am = np.mean(p110a)
p111am = np.mean(p111a)
p112am = np.mean(p112a)
p113am = np.mean(p113a)
p115am = np.mean(p115a)
p11xam = np.mean(h11xa)

h110as = np.std(h110a)
h111as = np.std(h111a)
h112as = np.std(h112a)
h113as = np.std(h113a)
h115as = np.std(h115a)
h11xas = np.std(h11xa)
p110as = np.std(p110a)
p111as = np.std(p111a)
p112as = np.std(p112a)
p113as = np.std(p113a)
p115as = np.std(p115a)
p11xas = np.std(h11xa)





### RUN 2
h110b = np.array([2.5, 2.4, 2.0, 2.0, 2.1])
h111b = np.array([2.4, 2.6, 2.1, 2.2, 2.3])
h112b = np.array([2.7, 2.5, 2.4, 2.4, 2.1])
h113b = np.array([1.9, 1.7, 1.5, 1.2, 1.3])
h115b = np.array([1.6, 1.6, 1.4, 1.1, 1.3])
h11xb = np.array([2.0, 1.5, 2.0, 1.4, 1.2])
p110b = np.array([2.8, 2.8, 2.4, 2.2, 2.5])
p111b = np.array([1.9, 2.0, 2.0, 1.3, 1.8])
p112b = np.array([2.6, 1.7, 1.7, 1.6, 2.3])
p113b = np.array([1.3, 1.7, 1.3, 1.2, 1.2])
p115b = np.array([1.3, 1.7, 1.3, 0.9, 1.1])
p11xb = np.array([0.9, 1.0, 0.7, 0.5, 1.1])

h110bm = np.mean(h110b)
h111bm = np.mean(h111b)
h112bm = np.mean(h112b)
h113bm = np.mean(h113b)
h115bm = np.mean(h115b)
h11xbm = np.mean(h11xb)
p110bm = np.mean(p110b)
p111bm = np.mean(p111b)
p112bm = np.mean(p112b)
p113bm = np.mean(p113b)
p115bm = np.mean(p115b)
p11xbm = np.mean(h11xb)

h110bs = np.std(h110b)
h111bs = np.std(h111b)
h112bs = np.std(h112b)
h113bs = np.std(h113b)
h115bs = np.std(h115b)
h11xbs = np.std(h11xb)
p110bs = np.std(p110b)
p111bs = np.std(p111b)
p112bs = np.std(p112b)
p113bs = np.std(p113b)
p115bs = np.std(p115b)
p11xbs = np.std(h11xb)

### RUN 3 (Jenny)
h110c = np.array([2.39, 2.33, 2.70, 2.13, 2.81])
h111c = np.array([2.18, 2.42, 3.33, 2.55, 2.70])
h112c = np.array([2.44, 2.43, 2.29, 2.27, 1.89])
h113c = np.array([2.12, 2.03, 1.85, 1.84, 1.55])
h115c = np.array([1.45, 1.76, 1.96, 1.76, 1.85])
h11xc = np.array([1.28, 1.57, 1.73, 1.21, 1.60])
p110c = np.array([2.87, 2.66, 2.75, 2.39, 2.69])
p111c = np.array([2.51, 2.38, 2.35, 1.81, 2.05])
p112c = np.array([2.50, 2.44, 2.48, 2.21, 1.99])
p113c = np.array([1.65, 1.55, 1.63, 1.86, 1.14])
p115c = np.array([1.03, 1.51, 1.32, 1.27, 1.29])
p11xc = np.array([0.62, 0.53, 0.86, 0.88, 0.88])

h110cm = np.mean(h110c)
h111cm = np.mean(h111c)
h112cm = np.mean(h112c)
h113cm = np.mean(h113c)
h115cm = np.mean(h115c)
h11xcm = np.mean(h11xc)
p110cm = np.mean(p110c)
p111cm = np.mean(p111c)
p112cm = np.mean(p112c)
p113cm = np.mean(p113c)
p115cm = np.mean(p115c)
p11xcm = np.mean(h11xc)

h110cs = np.std(h110c)
h111cs = np.std(h111c)
h112cs = np.std(h112c)
h113cs = np.std(h113c)
h115cs = np.std(h115c)
h11xcs = np.std(h11xc)
p110cs = np.std(p110c)
p111cs = np.std(p111c)
p112cs = np.std(p112c)
p113cs = np.std(p113c)
p115cs = np.std(p115c)
p11xcs = np.std(h11xc)

h11am = np.array([h110am, h111am, h112am, h113am, h115am, h11xam])/h110am*100
p11am = np.array([p110am, p111am, p112am, p113am, p115am, p11xam])/p110am*100
h11as = np.array([h110as, h111as, h112as, h113as, h115as, h11xas])/h110am*100
p11as = np.array([p110as, p111as, p112as, p113as, p115as, p11xas])/p110am*100

h11bm = np.array([h110bm, h111bm, h112bm, h113bm, h115bm, h11xbm])/h110bm*100
p11bm = np.array([p110bm, p111bm, p112bm, p113bm, p115bm, p11xbm])/p110bm*100
h11bs = np.array([h110bs, h111bs, h112bs, h113bs, h115bs, h11xbs])/h110bm*100
p11bs = np.array([p110bs, p111bs, p112bs, p113bs, p115bs, p11xbs])/p110bm*100

h11cm = np.array([h110cm, h111cm, h112cm, h113cm, h115cm, h11xcm])/h110cm*100
p11cm = np.array([p110cm, p111cm, p112cm, p113cm, p115cm, p11xcm])/p110cm*100
h11cs = np.array([h110cs, h111cs, h112cs, h113cs, h115cs, h11xcs])/h110cm*100
p11cs = np.array([p110cs, p111cs, p112cs, p113cs, p115cs, p11xcs])/p110cm*100

print(h11am)
print(h11as)
FS = 17
fig, ax = plt.subplots(2,3, figsize=(14,8.5),sharey="all")
ax[0,0].errorbar(D, h11am, yerr=h11as, uplims=True, lolims=True, label="T47D")
ax[0,0].errorbar(D, p11am, yerr=p11as, uplims=True, lolims=True, label="T47D-P")
ax[0,0].text(0.05, 0.95, "h11, a, april 21 (Henrik) ", transform=ax[0,0].transAxes)
ax[0,1].errorbar(D, h11bm, yerr=h11bs, uplims=True, lolims=True, label="T47D")
ax[0,1].errorbar(D, p11bm, yerr=p11bs, uplims=True, lolims=True, label="T47D-P")
ax[0,1].text(0.05, 0.95, "h11, b, sept. 21 (Henrik)", transform=ax[0,1].transAxes)
ax[0,2].errorbar(D, h11cm, yerr=h11cs, uplims=True, lolims=True, label="T47D")
ax[0,2].errorbar(D, p11cm, yerr=p11cs, uplims=True, lolims=True, label="T47D-P")
ax[0,2].text(0.2, 0.95, "h11, c, sept. (Jenny)", transform=ax[0,2].transAxes)
for i in range(2):
    for j in range(3):
        ax[i,j].legend(fontsize=FS-2)
        ax[i,j].set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax[i,j].set_xlabel("Dosis (Gy)",fontsize=FS)
        ax[i,j].tick_params(axis='both', which='major', labelsize=FS)
        ax[i,0].set_ylabel("Mitotisk index (%)",fontsize=FS)
plt.tight_layout()
plt.savefig("t47d_p_mitoticindex.pdf")
plt.show()
