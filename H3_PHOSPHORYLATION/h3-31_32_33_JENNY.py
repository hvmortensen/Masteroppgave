import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'cm'
mpl.rcParams['font.sans-serif'] = 'Times New Roman'
mpl.rcParams['font.family'] = 'Times New Roman'

D = np.array([0.0, 0.1, 0.2, 0.3, 0.5])

H31 = np.loadtxt("PQML_H3-31.txt")
H32 = np.loadtxt("PQML_H3-32.txt")
H33 = np.loadtxt("PQML_H3-33.txt")



a31 = np.zeros(5)
sa31 = np.zeros(5)
ap31 = np.zeros(5)
aps31 = np.zeros(5)
at31 = np.zeros(5)
ats31 = np.zeros(5)

a32 = np.zeros(5)
sa32 = np.zeros(5)
ap32 = np.zeros(5)
aps32 = np.zeros(5)
at32 = np.zeros(5)
ats32 = np.zeros(5)

a33 = np.zeros(5)
sa33 = np.zeros(5)
ap33 = np.zeros(5)
aps33 = np.zeros(5)
at33 = np.zeros(5)
ats33 = np.zeros(5)


for i in range(5):

    a31[i] = np.mean(H31[i])/   np.mean(H31[0])
    ap31[i] = np.mean(H31[i+5])/ np.mean(H31[5])
    at31[i] = np.mean(H31[i+10])/np.mean(H31[10])
    sa31[i] =  np.std(H31[i])/   np.mean(H31[0])
    aps31[i] =  np.std(H31[i+5])/ np.mean(H31[5])
    ats31[i] =  np.std(H31[i+10])/np.mean(H31[10])

    a32[i] = np.mean(H32[i])/   np.mean(H32[0])
    ap32[i] = np.mean(H32[i+5])/ np.mean(H32[5])
    at32[i] = np.mean(H32[i+10])/np.mean(H32[10])
    sa32[i] =  np.std(H32[i])/   np.mean(H32[0])
    aps32[i] =  np.std(H32[i+5])/ np.mean(H32[5])
    ats32[i] =  np.std(H32[i+10])/np.mean(H32[10])

    a33[i] = np.mean(H33[i])/   np.mean(H33[0])
    ap33[i] = np.mean(H33[i+5])/ np.mean(H33[5])
    at33[i] = np.mean(H33[i+10])/np.mean(H33[10])
    sa33[i] =  np.std(H33[i])/   np.mean(H33[0])
    aps33[i] =  np.std(H33[i+5])/ np.mean(H33[5])
    ats33[i] =  np.std(H33[i+10])/np.mean(H33[10])
#


FS = 17 + 5
fig, ax = plt.subplots(1,3, figsize=(14,4.5),sharey="all")
ax[0].errorbar(D, a31*100,  yerr=sa31*100,  uplims=True, lolims=True, label="T47D")
ax[0].errorbar(D, ap31*100, yerr=aps31*100, uplims=True, lolims=True, label="T47D-P")
ax[0].errorbar(D, at31*100, yerr=ats31*100, uplims=True, lolims=True, label="T47D-T")
ax[0].text(0.03, 0.93, "H3-31", transform=ax[0].transAxes,fontsize=FS-5)
ax[1].errorbar(D, a32*100,  yerr=sa32*100,  uplims=True, lolims=True, label="T47D")
ax[1].errorbar(D, ap32*100, yerr=aps32*100, uplims=True, lolims=True, label="T47D-P")
ax[1].errorbar(D, at32*100, yerr=ats32*100, uplims=True, lolims=True, label="T47D-T")
ax[1].text(0.03, 0.93, "H3-32", transform=ax[1].transAxes,fontsize=FS-5)
ax[2].errorbar(D, a33*100,  yerr=sa33*100,  uplims=True, lolims=True, label="T47D")
ax[2].errorbar(D, ap33*100, yerr=aps33*100, uplims=True, lolims=True, label="T47D-P")
ax[2].errorbar(D, at33*100, yerr=ats33*100, uplims=True, lolims=True, label="T47D-T")
ax[2].text(0.03, 0.93, "H3-33      (30% etanol) ", transform=ax[2].transAxes,fontsize=FS-5)

for i in range(3):
    ax[2].legend(fontsize=FS-2,loc=4)
    ax[i].set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax[i].set_xlabel("Dosis (Gy)",fontsize=FS)
    ax[i].tick_params(axis='both', which='major', labelsize=FS)
    ax[0].set_ylabel("Mitotisk index (%)",fontsize=FS)
plt.tight_layout()
plt.savefig("H3-tgfbeta3_JENNY.pdf")
plt.show()
# # print(a)
# plt.plot(D,a27)
# plt.plot(D,ap27)
# plt.plot(D,at27)
# plt.show()
#
# plt.plot(D,a28)
# plt.plot(D,ap28)
# plt.plot(D,at28)
# plt.show()
