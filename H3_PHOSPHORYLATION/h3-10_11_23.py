import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'cm'
mpl.rcParams['font.sans-serif'] = 'Times New Roman'
mpl.rcParams['font.family'] = 'Times New Roman'


D = np.array([0.0, 0.1, 0.2, 0.3, 0.5, 1.0])

H10 = np.loadtxt("H3-10_data.txt")
H11 = np.loadtxt("H3-11_henrik.txt")
H23 = np.loadtxt("H3-23_data.txt")

h10 = np.zeros(12) # means af hver linje hele filen
h11 = np.zeros(12)
h23 = np.zeros(12)
s10 = np.zeros(12) # std af hver linje i hele filen
s11 = np.zeros(12)
s23 = np.zeros(12)
for i in range(12):
    h10[i] = np.mean(H10[i])
    h11[i] = np.mean(H11[i])
    h23[i] = np.mean(H23[i])
    s10[i] = np.std(H10[i])
    s11[i] = np.std(H11[i])
    s23[i] = np.std(H23[i])

h10m = np.zeros(6) # første 6 normaliserede means
p10m = np.zeros(6) # næste 6 normaliserede means
h10s = np.zeros(6) # første 6 normaliserede std
p10s = np.zeros(6) # næste 6 normaliserede std
h11m = np.zeros(6)
p11m = np.zeros(6)
h11s = np.zeros(6)
p11s = np.zeros(6)
h23m = np.zeros(6)
p23m = np.zeros(6)
h23s = np.zeros(6)
p23s = np.zeros(6)
for i in range(6):
    h10m[i] = h10[i]/h10[0]
    p10m[i] = h10[i+6]/h10[6]
    h10s[i] = s10[i]/h10[0]
    p10s[i] = s10[i+6]/h10[6]
    h11m[i] = h11[i]/h11[0]
    p11m[i] = h11[i+6]/h11[6]
    h11s[i] = s11[i]/h11[0]
    p11s[i] = s11[i+6]/h11[6]
    h23m[i] = h23[i]/h23[0]
    p23m[i] = h23[i+6]/h23[6]
    h23s[i] = s23[i]/h23[0]
    p23s[i] = s23[i+6]/h23[6]

H = np.zeros(6)
P = np.zeros(6)
HS = np.zeros(6)
PS = np.zeros(6)
for i in range(6):
    H[i] = np.mean([h10m[i], h11m[i], h23m[i]])
    HS[i] = np.std([h10m[i], h11m[i], h23m[i]])/np.sqrt(3)
    P[i] = np.mean([p10m[i], p11m[i], p23m[i]])
    PS[i] = np.std([p10m[i], p11m[i], p23m[i]])/np.sqrt(3)

FS = 17+5
fig, ax = plt.subplots(1,3, figsize=(14,4.5),sharey="all")
ax[0].errorbar(D, h10m, yerr=h10s, uplims=True, lolims=True)
ax[0].errorbar(D, p10m, yerr=p10s, uplims=True, lolims=True)
ax[0].text(0.05, 0.95, "H3-10", transform=ax[0].transAxes,fontsize=FS-7)
ax[1].errorbar(D, h11m, yerr=h11s, uplims=True, lolims=True)
ax[1].errorbar(D, p11m, yerr=p11s, uplims=True, lolims=True)
ax[1].text(0.2, 0.95, "H3-11", transform=ax[1].transAxes,fontsize=FS-7)
ax[2].errorbar(D, h23m, yerr=h23s, uplims=True, lolims=True, label="T47D")
ax[2].errorbar(D, p23m, yerr=p23s, uplims=True, lolims=True, label="T47D-P")
ax[2].text(0.05, 0.95, "H3-23", transform=ax[2].transAxes,fontsize=FS-7)
ax[2].legend(fontsize=FS-2)

for i in range(3):
    ax[i].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # ax[i].set_yticks([0.4, 0.6, 0.8, 1.0, 1.2])

    ax[i].set_xlabel("Dosis (Gy)",fontsize=FS)
    ax[i].tick_params(axis='both', which='major', labelsize=FS)
    ax[0].set_ylabel("Mitotisk index (%)",fontsize=FS)
plt.tight_layout()
plt.savefig("H3-10_11_23all3.pdf")
plt.show()


FS = 17 
plt.errorbar(D, H, yerr=HS, uplims=True, lolims=True, label="T47D")
plt.errorbar(D, P, yerr=PS, uplims=True, lolims=True, label="T47D-P")
plt.xlabel("Dosis (Gy)",fontsize=FS)
plt.ylabel("Mitotisk indeks (%)",fontsize=FS)
plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
plt.tick_params(axis='both', which='major', labelsize=FS)
plt.legend(fontsize=FS)
plt.tight_layout()
plt.savefig("H3-10_11_23.pdf")
plt.show()
