import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'cm'
mpl.rcParams['font.sans-serif'] = 'Times New Roman'
mpl.rcParams['font.family'] = 'Times New Roman'

D = np.array([0.0, 0.1, 0.2, 0.3, 0.5])

H27 = np.loadtxt("PQML_H3-27.txt")
H28 = np.loadtxt("PQML_H3-28.txt")
H29 = np.loadtxt("PQML_H3-29.txt")
H30 = np.loadtxt("PQML_H3-30.txt")
H31 = np.loadtxt("PQML_H3-31.txt")
H32 = np.loadtxt("PQML_H3-32.txt")

a27 = np.zeros(5)
sa27 = np.zeros(5)
ap27 = np.zeros(5)
aps27 = np.zeros(5)
at27 = np.zeros(5)
ats27 = np.zeros(5)

a28 = np.zeros(5)
sa28 = np.zeros(5)
ap28 = np.zeros(5)
aps28 = np.zeros(5)
at28 = np.zeros(5)
ats28 = np.zeros(5)

a29 = np.zeros(5)
sa29 = np.zeros(5)
ap29 = np.zeros(5)
aps29 = np.zeros(5)
at29 = np.zeros(5)
ats29 = np.zeros(5)

a30 = np.zeros(5)
sa30 = np.zeros(5)
ap30 = np.zeros(5)
aps30 = np.zeros(5)
at30 = np.zeros(5)
ats30 = np.zeros(5)

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


for i in range(5):
    a27[i] = np.mean(H27[i])/np.mean(H27[0])
    ap27[i] = np.mean(H27[i+5])/np.mean(H27[5])
    at27[i] = np.mean(H27[i+10])/np.mean(H27[10])
    sa27[i] = np.std(H27[i])/np.mean(H27[0])
    aps27[i] = np.std(H27[i+5])/np.mean(H27[5])
    ats27[i] = np.std(H27[i+10])/np.mean(H27[10])

    a28[i] = np.mean(H28[i])/np.mean(H28[0])
    ap28[i] = np.mean(H28[i+5])/np.mean(H28[5])
    at28[i] = np.mean(H28[i+10])/np.mean(H28[10])
    sa28[i] = np.std(H28[i])/np.mean(H28[0])
    aps28[i] = np.std(H28[i+5])/np.mean(H28[5])
    ats28[i] = np.std(H28[i+10])/np.mean(H28[10])

    a29[i] = np.mean(H29[i])/np.mean(H29[0])
    ap29[i] = np.mean(H29[i+5])/np.mean(H29[5])
    at29[i] = np.mean(H29[i+10])/np.mean(H29[10])
    sa29[i] = np.std(H29[i])/np.mean(H29[0])
    aps29[i] = np.std(H29[i+5])/np.mean(H29[5])
    ats29[i] = np.std(H29[i+10])/np.mean(H29[10])

    a30[i] = np.mean(H30[i])/np.mean(H30[0])
    ap30[i] = np.mean(H30[i+5])/np.mean(H30[5])
    at30[i] = np.mean(H30[i+10])/np.mean(H30[10])
    sa30[i] = np.std(H30[i])/np.mean(H30[0])
    aps30[i] = np.std(H30[i+5])/np.mean(H30[5])
    ats30[i] = np.std(H30[i+10])/np.mean(H30[10])

    a31[i] = np.mean(H31[i])/   np.mean(H31[0])
    ap31[i] = np.mean(H31[i+5])/ np.mean(H31[5])
    at31[i] = np.mean(H31[i+10])/np.mean(H31[10])
    sa31[i]  = np.std(H31[i])/   np.mean(H31[0])
    aps31[i]  = np.std(H31[i+5])/ np.mean(H31[5])
    ats31[i]  = np.std(H31[i+10])/np.mean(H31[10])

    a32[i] = np.mean(H32[i])/   np.mean(H32[0])
    ap32[i] = np.mean(H32[i+5])/ np.mean(H32[5])
    at32[i] = np.mean(H32[i+10])/np.mean(H32[10])
    sa32[i]  = np.std(H32[i])/   np.mean(H32[0])
    aps32[i]  = np.std(H32[i+5])/ np.mean(H32[5])
    ats32[i]  = np.std(H32[i+10])/np.mean(H32[10])

#


FS = 17 + 5
fig, ax = plt.subplots(2,3, figsize=(14,8.5),sharey="all")
ax[0,0].errorbar(D, a27*100,  yerr=sa27*100,  uplims=True, lolims=True, label="T47D")
ax[0,0].errorbar(D, ap27*100, yerr=aps27*100, uplims=True, lolims=True, label="T47D-P")
ax[0,0].errorbar(D, at27*100, yerr=ats27*100, uplims=True, lolims=True, label="T47D-T")
ax[0,0].text(0.05, 0.93, "H3-27", transform=ax[0,0].transAxes,fontsize=FS-5)
ax[0,1].errorbar(D, a28*100,  yerr=sa28*100,  uplims=True, lolims=True, label="T47D")
ax[0,1].errorbar(D, ap28*100, yerr=aps28*100, uplims=True, lolims=True, label="T47D-P")
ax[0,1].errorbar(D, at28*100, yerr=ats28*100, uplims=True, lolims=True, label="T47D-T")
ax[0,1].text(0.05, 0.93, "H3-28", transform=ax[0,1].transAxes,fontsize=FS-5)
ax[0,2].errorbar(D, a29*100,  yerr=sa29*100,  uplims=True, lolims=True, label="T47D")
ax[0,2].errorbar(D, ap29*100, yerr=aps29*100, uplims=True, lolims=True, label="T47D-P")
ax[0,2].errorbar(D, at29*100, yerr=ats29*100, uplims=True, lolims=True, label="T47D-T")
ax[0,2].text(0.05, 0.93, "H3-29", transform=ax[0,2].transAxes,fontsize=FS-5)
ax[1,0].errorbar(D, a30*100,  yerr=sa30*100,  uplims=True, lolims=True, label="T47D")
ax[1,0].errorbar(D, ap30*100, yerr=aps30*100, uplims=True, lolims=True, label="T47D-P")
ax[1,0].errorbar(D, at30*100, yerr=ats30*100, uplims=True, lolims=True, label="T47D-T")
ax[1,0].text(0.05, 0.93, "H3-30", transform=ax[1,0].transAxes,fontsize=FS-5)
ax[1,1].errorbar(D, a31*100,  yerr=sa31*100,  uplims=True, lolims=True, label="T47D")
ax[1,1].errorbar(D, ap31*100, yerr=aps31*100, uplims=True, lolims=True, label="T47D-P")
ax[1,1].errorbar(D, at31*100, yerr=ats31*100, uplims=True, lolims=True, label="T47D-T")
ax[1,1].text(0.05, 0.93, "H3-31", transform=ax[1,1].transAxes,fontsize=FS-5)
ax[1,2].errorbar(D, a32*100,  yerr=sa32*100,  uplims=True, lolims=True, label="T47D")
ax[1,2].errorbar(D, ap32*100, yerr=aps32*100, uplims=True, lolims=True, label="T47D-P")
ax[1,2].errorbar(D, at32*100, yerr=ats32*100, uplims=True, lolims=True, label="T47D-T")
ax[1,2].text(0.05, 0.93, "H3-32", transform=ax[1,2].transAxes,fontsize=FS-5)


for i in range(2):
    for j in range(3):
        ax[1,2].legend(fontsize=FS-2,loc=1)
        ax[i,j].set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
        ax[i,j].set_xlabel("Dosis (Gy)",fontsize=FS)
        ax[i,j].tick_params(axis='both', which='major', labelsize=FS)
        ax[i,0].set_ylabel("Mitotisk index (%)",fontsize=FS)
plt.tight_layout()
plt.savefig("H3-tgfbeta3_exps_inklJENNY.pdf")
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
