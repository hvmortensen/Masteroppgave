import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'cm'
mpl.rcParams['font.sans-serif'] = 'Times New Roman'
mpl.rcParams['font.family'] = 'Times New Roman'


D = np.array([0.0, 0.1, 0.2, 0.3, 0.5])

H = np.loadtxt("H3-11_henrik.txt")
N = np.loadtxt("H3-11_nina.txt")
I = np.loadtxt("H3-11_ingunn.txt")
J = np.loadtxt("H3-11_jenny.txt")


D = np.array([0.0, 0.1, 0.2, 0.3, 0.5])

H = np.loadtxt("H3-18_henrik.txt")
N = np.loadtxt("H3-18_nina.txt")
I = np.loadtxt("H3-18_ingunn.txt")
J = np.loadtxt("H3-18_jenny.txt")

hm = np.zeros(15) #means af H-array
nm = np.zeros(15)
im = np.zeros(15)
jm = np.zeros(15)
hss = np.zeros(15) # std af H-array
nss = np.zeros(15)
iss = np.zeros(15)
jss = np.zeros(15)
for i in range(15):
    hm[i] = np.mean(H[i])
    nm[i] = np.mean(N[i])
    im[i] = np.mean(I[i])
    jm[i] = np.mean(J[i])
    hss[i] = np.std(H[i])
    nss[i] = np.std(N[i])
    iss[i] = np.std(I[i])
    jss[i] = np.std(J[i])


hmm = np.zeros(5) # normaliseret T47D
hpp = np.zeros(5) # normaliseret T47D-P
htt = np.zeros(5) # normaliseret T47D-T
hsss = np.zeros(5)# normaliseret standardafvig
hpss = np.zeros(5)
htss = np.zeros(5)

nmm = np.zeros(5)
npp = np.zeros(5)
ntt = np.zeros(5)
nsss = np.zeros(5)
npss = np.zeros(5)
ntss = np.zeros(5)
imm = np.zeros(5)
ipp = np.zeros(5)
itt = np.zeros(5)
isss = np.zeros(5)
ipss = np.zeros(5)
itss= np.zeros(5)
jmm = np.zeros(5)
jpp = np.zeros(5)
jtt = np.zeros(5)
jsss = np.zeros(5)
jpss = np.zeros(5)
jtss = np.zeros(5)

for i in range(5):
    hmm[i] = hm[i]/hm[0]
    hpp[i] = hm[i+5]/hm[5]
    htt[i] = hm[i+10]/hm[10]
    hsss[i] = hss[i]/hm[0]
    hpss[i] = hss[i+5]/hm[5]
    htss[i] = hss[i+10]/hm[10]

    nmm[i] = nm[i]/nm[0]
    npp[i] = nm[i+5]/nm[5]
    ntt[i] = nm[i+10]/nm[10]
    nsss[i] = nss[i]/nm[0]
    npss[i] = nss[i+5]/nm[5]
    ntss[i] = nss[i+10]/nm[10]

    imm[i] = im[i]/im[0]
    ipp[i] = im[i+5]/im[5]
    itt[i] = im[i+10]/im[10]
    isss[i] = iss[i]/im[0]
    ipss[i] = iss[i+5]/im[5]
    itss[i] = iss[i+10]/im[10]

    jmm[i] = jm[i]/jm[0]
    jpp[i] = jm[i+5]/jm[5]
    jtt[i] = jm[i+10]/jm[10]
    jsss[i] = jss[i]/jm[0]
    jpss[i] = jss[i+5]/jm[5]
    jtss[i] = jss[i+10]/jm[10]

M = np.zeros(5) # overordnet mean fra alle observatører
P = np.zeros(5)
T = np.zeros(5)
MS = np.zeros(5)# SEM
PS = np.zeros(5)
TS = np.zeros(5)

for i in range(5):
    M[i] = np.mean([hmm[i], nmm[i], imm[i], jmm[i]])
    P[i] = np.mean([hpp[i], npp[i], ipp[i], jpp[i]])
    T[i] = np.mean([htt[i], ntt[i], itt[i], jtt[i]])
    MS[i] = np.std([hmm[i], nmm[i], imm[i], jmm[i]])/np.sqrt(4)
    PS[i] = np.std([hpp[i], npp[i], ipp[i], jpp[i]])/np.sqrt(4)
    PS[i] = np.std([htt[i], ntt[i], itt[i], jtt[i]])/np.sqrt(4)

FS = 17
fig, ax = plt.subplots(1,4, figsize=(14,4.5),sharey="all")
ax[0].errorbar(D, hmm, yerr=hsss, uplims=True, lolims=True, label="T47D")
ax[0].errorbar(D, hpp, yerr=hpss, uplims=True, lolims=True, label="T47D-P")
ax[0].errorbar(D, htt, yerr=htss, uplims=True, lolims=True, label="T47D-T")
ax[0].text(0.05, 0.95, "Henrik", transform=ax[0].transAxes)
ax[1].errorbar(D, nmm, yerr=nsss, uplims=True, lolims=True, label="T47D")
ax[1].errorbar(D, npp, yerr=npss, uplims=True, lolims=True, label="T47D-P")
ax[1].errorbar(D, ntt, yerr=ntss, uplims=True, lolims=True, label="T47D-T")
ax[1].text(0.05, 0.95, "Nina", transform=ax[1].transAxes)
ax[2].errorbar(D, imm, yerr=isss, uplims=True, lolims=True, label="T47D")
ax[2].errorbar(D, ipp, yerr=ipss, uplims=True, lolims=True, label="T47D-P")
ax[2].errorbar(D, itt, yerr=itss, uplims=True, lolims=True, label="T47D-T")
ax[2].text(0.05, 0.95, "Ingunn", transform=ax[2].transAxes)
ax[3].errorbar(D, jmm, yerr=jsss, uplims=True, lolims=True, label="T47D")
ax[3].errorbar(D, jpp, yerr=jpss, uplims=True, lolims=True, label="T47D-P")
ax[3].errorbar(D, jtt, yerr=jtss, uplims=True, lolims=True, label="T47D-T")
ax[3].text(0.05, 0.95, "Jenny", transform=ax[3].transAxes)


for i in range(4):
    ax[i].legend(fontsize=FS-2)
    ax[i].set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax[i].set_xlabel("Dosis (Gy)",fontsize=FS)
    ax[i].tick_params(axis='both', which='major', labelsize=FS)
    ax[0].set_ylabel("Mitotisk index (%)",fontsize=FS)
plt.tight_layout()
plt.savefig("H3-18_unbiased_observers.pdf")
plt.show()

plt.errorbar(D, M, yerr=MS, uplims=True, lolims=True, label="T47D")
plt.errorbar(D, P, yerr=PS, uplims=True, lolims=True, label="T47D-P")
plt.errorbar(D, T, yerr=TS, uplims=True, lolims=True, label="T47D-T")
plt.xlabel("Dosis (Gy)",fontsize=FS)
plt.ylabel("Mitotisk indeks (%)",fontsize=FS)
plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
plt.tick_params(axis='both', which='major', labelsize=FS)
# plt.text(0.001, 110, "Gennemsnit")
plt.legend(fontsize=FS-2)
plt.tight_layout()
plt.savefig("H3-18_mean_unbiased.pdf")

plt.show()
