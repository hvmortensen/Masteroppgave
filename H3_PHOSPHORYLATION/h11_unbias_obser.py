import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'cm'
mpl.rcParams['font.sans-serif'] = 'Times New Roman'
mpl.rcParams['font.family'] = 'Times New Roman'

number_of_observers = 4
y = number_of_observers


H = np.loadtxt("H3-11_henrik.txt")
N = np.loadtxt("H3-11_nina.txt")
I = np.loadtxt("H3-11_ingunn.txt")
J = np.loadtxt("H3-11_jenny.txt")

D = np.array([0.0, 0.1, 0.2, 0.3, 0.5, 1.0])

x = len(D)
print(x)

hm = np.zeros(12)
nm = np.zeros(12)
im = np.zeros(12)
jm = np.zeros(12)
hss = np.zeros(12)
nss = np.zeros(12)
iss = np.zeros(12)
jss = np.zeros(12)


for i in range(12):
    hm[i] = np.mean(H[i])
    nm[i] = np.mean(N[i])
    im[i] = np.mean(I[i])
    jm[i] = np.mean(J[i])
    hss[i] = np.std(H[i])
    nss[i] = np.std(N[i])
    iss[i] = np.std(I[i])
    jss[i] = np.std(J[i])

hmm = np.zeros(6)
hpp = np.zeros(6)
hsss = np.zeros(6)
hpss = np.zeros(6)
nmm = np.zeros(6)
npp = np.zeros(6)
nsss = np.zeros(6)
npss = np.zeros(6)
imm = np.zeros(6)
ipp = np.zeros(6)
isss = np.zeros(6)
ipss = np.zeros(6)
jmm = np.zeros(6)
jpp = np.zeros(6)
jsss = np.zeros(6)
jpss = np.zeros(6)

for i in range(6):
    hmm[i] = hm[i]/hm[0]
    hpp[i] = hm[i+6]/hm[6]
    hsss[i] = hss[i]/hm[0]
    hpss[i] = hss[i+6]/hm[6]
    nmm[i] = nm[i]/nm[0]
    npp[i] = nm[i+6]/nm[6]
    nsss[i] = nss[i]/nm[0]
    npss[i] = nss[i+6]/nm[6]
    imm[i] = im[i]/im[0]
    ipp[i] = im[i+6]/im[6]
    isss[i] = iss[i]/im[0]
    ipss[i] = iss[i+6]/im[6]
    jmm[i] = jm[i]/jm[0]
    jpp[i] = jm[i+6]/jm[6]
    jsss[i] = jss[i]/jm[0]
    jpss[i] = jss[i+6]/jm[6]

M = np.zeros(6)
P = np.zeros(6)
MS = np.zeros(6)
PS = np.zeros(6)

for i in range(6):
    M[i] = np.mean([hmm[i], nmm[i], imm[i], jmm[i]])
    MS[i] = np.std([hmm[i], nmm[i], imm[i], jmm[i]])/np.sqrt(4)
    P[i] = np.mean([hpp[i], npp[i], ipp[i], jpp[i]])
    PS[i] = np.std([hpp[i], npp[i], ipp[i], jpp[i]])/np.sqrt(4)


FS = 17+5
fig, ax = plt.subplots(1,4, figsize=(14,4.5),sharey="all")
ax[0].errorbar(D, hmm, yerr=hsss, uplims=True, lolims=True)#, label="T47D")
ax[0].errorbar(D, hpp, yerr=hpss, uplims=True, lolims=True)#, label="T47D-P")
ax[0].text(0.05, 0.93, "Henrik", transform=ax[0].transAxes,fontsize=FS-5)
ax[1].errorbar(D, nmm, yerr=nsss, uplims=True, lolims=True)#, label="T47D")
ax[1].errorbar(D, npp, yerr=npss, uplims=True, lolims=True)#, label="T47D-P")
ax[1].text(0.05, 0.93, "Nina", transform=ax[1].transAxes,fontsize=FS-5)
ax[2].errorbar(D, imm, yerr=isss, uplims=True, lolims=True)#, label="T47D")
ax[2].errorbar(D, ipp, yerr=ipss, uplims=True, lolims=True)#, label="T47D-P")
ax[2].text(0.2, 0.93, "Ingunn", transform=ax[2].transAxes,fontsize=FS-5)
ax[3].errorbar(D, jmm, yerr=jsss, uplims=True, lolims=True, label="T47D")
ax[3].errorbar(D, jpp, yerr=jpss, uplims=True, lolims=True, label="T47D-P")
ax[3].text(0.05, 0.93, "Jenny", transform=ax[3].transAxes,fontsize=FS-5)


for i in range(4):
    ax[3].legend(fontsize=FS-2)
    ax[i].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax[i].set_xlabel("Dosis (Gy)",fontsize=FS)
    ax[i].tick_params(axis='both', which='major', labelsize=FS)
    ax[0].set_ylabel("Mitotisk index (%)",fontsize=FS)
plt.tight_layout()
plt.savefig("H3-11_unbiased_observers.pdf")
plt.show()

plt.errorbar(D, M, yerr=MS, uplims=True, lolims=True, label="T47D")
plt.errorbar(D, P, yerr=PS, uplims=True, lolims=True, label="T47D-P")
plt.xlabel("Dosis (Gy)",fontsize=FS)
plt.ylabel("Mitotisk indeks (%)",fontsize=FS)
plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
plt.tick_params(axis='both', which='major', labelsize=FS)
# plt.text(0.001, 110, "Gennemsnit")
plt.legend(fontsize=FS-2)
plt.tight_layout()
plt.savefig("H3-11_mean_unbiased.pdf")

plt.show()
