import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'cm'
mpl.rcParams['font.sans-serif'] = 'Times New Roman'
mpl.rcParams['font.family'] = 'Times New Roman'

D = np.array([0, 0.1, 0.2, 0.3, 0.5])

H3_18 = np.loadtxt("H3-18_data.txt")
H3_20 = np.loadtxt("H3-20_data.txt")
M = H3_20#

ah = M[0:15][0:5]   #første 5 rækker af de første 15 rækker
ap = M[0:15][5:10]  #næste 5 rækker af de første 15 rækker
at = M[0:15][10:15] #sidste 5 rækker af de første 15 rækker

bh = M[15:30][0:5]  #første 5 rækker af de næste 15 rækker
bp = M[15:30][5:10] #næste 5 rækker af de næste 15 rækker
bt = M[15:30][10:15]#sidste 5 rækker af de næste 15 rækker

ch = M[30:45][0:5]  #første 5 rækker af de sidste 15 rækker
cp = M[30:45][5:10] #næste 5 rækker af de sidste 15 rækker
ct = M[30:45][10:15]#sidste 5 rækker af de sidste 15 rækker

print(ah)

ahm = np.zeros(M.shape[1])
apm = np.zeros(M.shape[1])
atm = np.zeros(M.shape[1])
ahs = np.zeros(M.shape[1])
aps = np.zeros(M.shape[1])
ats = np.zeros(M.shape[1])

bhm = np.zeros(M.shape[1])
bpm = np.zeros(M.shape[1])
btm = np.zeros(M.shape[1])
bhs = np.zeros(M.shape[1])
bps = np.zeros(M.shape[1])
bts = np.zeros(M.shape[1])

chm = np.zeros(M.shape[1])
cpm = np.zeros(M.shape[1])
ctm = np.zeros(M.shape[1])
chs = np.zeros(M.shape[1])
cps = np.zeros(M.shape[1])
cts = np.zeros(M.shape[1])

for i in range(5):
    ahm[i] = np.mean(ah[i])/np.mean(ah[0])*100
    apm[i] = np.mean(ap[i])/np.mean(ap[0])*100
    atm[i] = np.mean(at[i])/np.mean(at[0])*100
    ahs[i] = np.std(ah[i])/np.mean(ah[0])*100
    aps[i] = np.std(ap[i])/np.mean(ap[0])*100
    ats[i] = np.std(at[i])/np.mean(at[0])*100

    bhm[i] = np.mean(bh[i])/np.mean(bh[0])*100
    bpm[i] = np.mean(bp[i])/np.mean(bp[0])*100
    btm[i] = np.mean(bt[i])/np.mean(bt[0])*100
    bhs[i] = np.std(bh[i])/np.mean(bh[0])*100
    bps[i] = np.std(bp[i])/np.mean(bp[0])*100
    bts[i] = np.std(bt[i])/np.mean(bt[0])*100

    chm[i] = np.mean(ch[i])/np.mean(ch[0])*100
    cpm[i] = np.mean(cp[i])/np.mean(cp[0])*100
    ctm[i] = np.mean(ct[i])/np.mean(ct[0])*100
    chs[i] = np.std(ch[i])/np.mean(ch[0])*100
    cps[i] = np.std(cp[i])/np.mean(cp[0])*100
    cts[i] = np.std(ct[i])/np.mean(ct[0])*100

print(ahm)
print(apm)
print(atm)
print(ahs)
print(aps)
print(ats)

hm = np.zeros(M.shape[1])
pm = np.zeros(M.shape[1])
tm = np.zeros(M.shape[1])
hs = np.zeros(M.shape[1])
ps = np.zeros(M.shape[1])
ts = np.zeros(M.shape[1])

for i in range(5):
    hm[i] = np.mean([ahm[i], bhm[i], chm[i]])
    pm[i] = np.mean([apm[i], bpm[i], cpm[i]])
    tm[i] = np.mean([atm[i], btm[i], ctm[i]])
    hs[i] = np.std([ahs[i], bhs[i], chs[i]])/np.sqrt(3)
    ps[i] = np.std([aps[i], bps[i], cps[i]])/np.sqrt(3)
    ts[i] = np.std([ats[i], bts[i], cts[i]])/np.sqrt(3)


# h18m = np.zeros(M.shape[1])
# p18m = np.zeros(M.shape[1])
# t18m = np.zeros(M.shape[1])
# h18s = np.zeros(M.shape[1])
# p18s = np.zeros(M.shape[1])
# t18s = np.zeros(M.shape[1])
if M.all==H3_18.all:
    h18m = hm
    p18m = pm
    t18m = tm
    h18s = hs
    p18s = ps
    t18s = ts

if M.all==H3_20.all:
    h20m = hm
    p20m = pm
    t20m = tm
    h20s = hs
    p20s = ps
    t20s = ts



FS = 17
fig, ax = plt.subplots(1,3, figsize=(14,4.5),sharey="all")
ax[0].errorbar(D, ahm, yerr=ahs, uplims=True, lolims=True, label="T47D")
ax[0].errorbar(D, apm, yerr=aps, uplims=True, lolims=True, label="T47D-P")
ax[0].errorbar(D, atm, yerr=ats, uplims=True, lolims=True, label="T47D-T")
ax[0].text(0.05, 0.95, "Første gating", transform=ax[0].transAxes)
ax[1].errorbar(D, bhm, yerr=bhs, uplims=True, lolims=True, label="T47D")
ax[1].errorbar(D, bpm, yerr=bps, uplims=True, lolims=True, label="T47D-P")
ax[1].errorbar(D, btm, yerr=bts, uplims=True, lolims=True, label="T47D-T")
ax[1].text(0.05, 0.95, "Anden gating", transform=ax[1].transAxes)
ax[2].errorbar(D, chm, yerr=chs, uplims=True, lolims=True, label="T47D")
ax[2].errorbar(D, cpm, yerr=cps, uplims=True, lolims=True, label="T47D-P")
ax[2].errorbar(D, ctm, yerr=cts, uplims=True, lolims=True, label="T47D-T")
ax[2].text(0.05, 0.95, "Tredje gating", transform=ax[2].transAxes)
for i in range(3):
    ax[i].legend(fontsize=FS-2)
    ax[i].set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax[i].set_xlabel("Dosis (Gy)",fontsize=FS)
    ax[i].tick_params(axis='both', which='major', labelsize=FS)
    ax[0].set_ylabel("Mitotisk index (%)",fontsize=FS)
plt.tight_layout()
# plt.savefig("tgfbeta_mitoticindex.pdf")
plt.show()

plt.errorbar(D, hm, yerr=hs, uplims=True, lolims=True, label="T47D")
plt.errorbar(D, pm, yerr=ps, uplims=True, lolims=True, label="T47D-P")
plt.errorbar(D, tm, yerr=ts, uplims=True, lolims=True, label="T47D_T")
plt.xlabel("Dosis (Gy)",fontsize=FS)
plt.ylabel("Mitotisk indeks (%)",fontsize=FS)
plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
plt.tick_params(axis='both', which='major', labelsize=FS)
plt.text(0.001, 110, "Gennemsnit")
plt.legend(fontsize=FS-2)
plt.tight_layout()
# plt.savefig("mitoticindex.pdf")

plt.show()
