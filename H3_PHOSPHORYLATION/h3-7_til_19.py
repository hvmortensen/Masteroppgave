import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'cm'
mpl.rcParams['font.sans-serif'] = 'Times New Roman'
mpl.rcParams['font.family'] = 'Times New Roman'

D = np.array([0.0, 0.1, 0.2, 0.3, 0.5, 1.0])

H7 = np.loadtxt("PQML_H3-7.txt")
H9 = np.loadtxt("PQML_H3-9.txt")
H10 = np.loadtxt("PQML_H3-10.txt")
H11 = np.loadtxt("PQML_H3-11.txt")
H13 = np.loadtxt("PQML_H3-13.txt")
H14 = np.loadtxt("PQML_H3-14.txt")
H15 = np.loadtxt("PQML_H3-15.txt")
H23 = np.loadtxt("PQML_H3-23.txt")

a7 = np.zeros(6)
b7 = np.zeros(6)
c7 = np.zeros(6)
d7 = np.zeros(6)

a9 = np.zeros(6)
b9 = np.zeros(6)
c9 = np.zeros(6)
d9 = np.zeros(6)

a10 = np.zeros(6)
b10 = np.zeros(6)
c10 = np.zeros(6)
d10 = np.zeros(6)

a11 = np.zeros(6)
b11 = np.zeros(6)
c11 = np.zeros(6)
d11 = np.zeros(6)

a13 = np.zeros(6)
b13 = np.zeros(6)
c13 = np.zeros(6)
d13 = np.zeros(6)

a14 = np.zeros(6)
b14 = np.zeros(6)
c14 = np.zeros(6)
d14 = np.zeros(6)

a15 = np.zeros(6)
b15 = np.zeros(6)
c15 = np.zeros(6)
d15 = np.zeros(6)

a23 = np.zeros(6)
b23 = np.zeros(6)
c23 = np.zeros(6)
d23 = np.zeros(6)

for i in range(6):
    a7[i] = np.mean(H7[i])/np.mean(H7[0])
    c7[i] = np.mean(H7[i+6])/np.mean(H7[6])
    b7[i] = np.std(H7[i])/np.mean(H7[0])
    d7[i] = np.std(H7[i+6])/np.mean(H7[6])

    a9[i] = np.mean(H9[i])/  np.mean(H9[0])
    c9[i] = np.mean(H9[i+6])/np.mean(H9[6])
    b9[i] =  np.std(H9[i])/  np.mean(H9[0])
    d9[i] =  np.std(H9[i+6])/np.mean(H9[6])

    a10[i] = np.mean(H10[i])/  np.mean(H10[0])
    c10[i] = np.mean(H10[i+6])/np.mean(H10[6])
    b10[i] =  np.std(H10[i])/  np.mean(H10[0])
    d10[i] =  np.std(H10[i+6])/np.mean(H10[6])

    a11[i] = np.mean(H11[i])/  np.mean(H11[0])
    c11[i] = np.mean(H11[i+6])/np.mean(H11[6])
    b11[i] =  np.std(H11[i])/  np.mean(H11[0])
    d11[i] =  np.std(H11[i+6])/np.mean(H11[6])

    a13[i] = np.mean(H13[i])/  np.mean(H13[0])
    c13[i] = np.mean(H13[i+6])/np.mean(H13[6])
    b13[i] =  np.std(H13[i])/  np.mean(H13[0])
    d13[i] =  np.std(H13[i+6])/np.mean(H13[6])

    a14[i] = np.mean(H14[i])/  np.mean(H14[0])
    c14[i] = np.mean(H14[i+6])/np.mean(H14[6])
    b14[i] =  np.std(H14[i])/  np.mean(H14[0])
    d14[i] =  np.std(H14[i+6])/np.mean(H14[6])

    a15[i] = np.mean(H15[i])/  np.mean(H15[0])
    c15[i] = np.mean(H15[i+6])/np.mean(H15[6])
    b15[i] =  np.std(H15[i])/  np.mean(H15[0])
    d15[i] =  np.std(H15[i+6])/np.mean(H15[6])

    a23[i] = np.mean(H23[i])/  np.mean(H23[0])
    c23[i] = np.mean(H23[i+6])/np.mean(H23[6])
    b23[i] =  np.std(H23[i])/  np.mean(H23[0])
    d23[i] =  np.std(H23[i+6])/np.mean(H23[6])

FS = 17 + 5
fig, ax = plt.subplots(2,4, figsize=(14,8.5),sharey="all")
ax[0,0].errorbar(D, a7*100,  yerr=b7*100,  uplims=True, lolims=True, label="T47D")
ax[0,0].errorbar(D, c7*100, yerr=d7*100, uplims=True, lolims=True, label="T47D-P")
ax[0,0].text(0.05, 0.93, "H3-7", transform=ax[0,0].transAxes,fontsize=FS-5)
ax[0,1].errorbar(D, a9*100,  yerr=b9*100,  uplims=True, lolims=True, label="T47D")
ax[0,1].errorbar(D, c9*100, yerr=d9*100, uplims=True, lolims=True, label="T47D-P")
ax[0,1].text(0.05, 0.93, "H3-9", transform=ax[0,1].transAxes,fontsize=FS-5)
ax[0,2].errorbar(D, a10*100,  yerr=b10*100,  uplims=True, lolims=True, label="T47D")
ax[0,2].errorbar(D, c10*100, yerr=d10*100, uplims=True, lolims=True, label="T47D-P")
ax[0,2].text(0.05, 0.93, "H3-10", transform=ax[0,2].transAxes,fontsize=FS-5)
ax[0,3].errorbar(D, a11*100,  yerr=b11*100,  uplims=True, lolims=True, label="T47D")
ax[0,3].errorbar(D, c11*100, yerr=d11*100, uplims=True, lolims=True, label="T47D-P")
ax[0,3].text(0.05, 0.93, "H3-11", transform=ax[0,3].transAxes,fontsize=FS-5)

ax[1,0].errorbar(D, a13*100,  yerr=b13*100,  uplims=True, lolims=True, label="T47D")
ax[1,0].errorbar(D, c13*100,  yerr=d13*100, uplims=True, lolims=True, label="T47D-P")
ax[1,0].text(0.05, 0.93, "H3-13", transform=ax[1,0].transAxes,fontsize=FS-5)
ax[1,1].errorbar(D, a14*100,  yerr=b14*100,  uplims=True, lolims=True, label="T47D")
ax[1,1].errorbar(D, c14*100,  yerr=d14*100, uplims=True, lolims=True, label="T47D-P")
ax[1,1].text(0.05, 0.93, "H3-14", transform=ax[1,1].transAxes,fontsize=FS-5)
ax[1,2].errorbar(D, a15*100,  yerr=b15*100,  uplims=True, lolims=True, label="T47D")
ax[1,2].errorbar(D, c15*100,  yerr=d15*100, uplims=True, lolims=True, label="T47D-P")
ax[1,2].text(0.3, 0.93, "H3-15", transform=ax[1,2].transAxes,fontsize=FS-5)
ax[1,3].errorbar(D, a23*100,  yerr=b23*100,  uplims=True, lolims=True, label="T47D")
ax[1,3].errorbar(D, c23*100,  yerr=d23*100, uplims=True, lolims=True, label="T47D-P")
ax[1,3].text(0.05, 0.93, "H3-23", transform=ax[1,3].transAxes,fontsize=FS-5)


for i in range(2):
    for j in range(4):
        ax[1,3].legend(fontsize=FS-2,loc=1)
        ax[i,j].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax[i,j].set_xlabel("Dosis (Gy)",fontsize=FS)
        ax[i,j].tick_params(axis='both', which='major', labelsize=FS)
        ax[i,0].set_ylabel("Mitotisk index (%)",fontsize=FS)
plt.tight_layout()
plt.savefig("H3-primed_exps.pdf")
plt.show()
