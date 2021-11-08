import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'cm'
mpl.rcParams['font.sans-serif'] = 'Times New Roman'
mpl.rcParams['font.family'] = 'Times New Roman'

number_of_observers = 2
no = number_of_observers


D = np.array([0.0, 0.1, 0.2, 0.3, 0.5])
dl = len(D)


number_of_celltypes = 3
nc = number_of_celltypes
h25 = np.loadtxt("H3-25_data.txt")
h26 = np.loadtxt("H3-26_data.txt")

M = np.zeros((no, nc, dl))
S = np.zeros((no, nc, dl))

for i in range(dl):
    for j in range(nc):
        M[0,j,i] = np.mean(h25[i + j*dl])/np.mean(h25[j*dl])
        M[1,j,i] = np.mean(h26[i + j*dl])/np.mean(h26[j*dl])
        S[0,j,i] = np.std(h25[i + j*dl])/np.mean(h25[j*dl])
        S[1,j,i] = np.std(h26[i + j*dl])/np.mean(h26[j*dl])



MM = np.zeros((nc,dl))      # Mitotic index Mean
MS = np.zeros((nc,dl))      # Mitotic index Standard deviation
for i in range(nc):
    for j in range(dl):
        MM[i,j] = np.mean([M[0,i,j], M[1,i,j]])
        MS[i,j] = np.std([M[0,i,j], M[1,i,j]])/np.sqrt(no)

FS = 17
fig, ax = plt.subplots(1,no, figsize=(14,4.5),sharey="all")
for i in range(no):
    for j in range(nc):
        ax[i].errorbar(D, M[i,j], yerr=S[i,j], uplims=True, lolims=True)
        ax[0].text(0.05, 0.95, "H3-25", transform=ax[0].transAxes)
        ax[1].text(0.05, 0.95, "H3-26", transform=ax[1].transAxes)
        ax[i].set_xlabel("Dosis (Gy)",fontsize=FS)
        ax[0].set_ylabel("Mitotisk index (%)",fontsize=FS)
        ax[i].tick_params(axis='both', which='major', labelsize=FS)
        ax[no-1].legend(["T47D", "T47D-P", "T47D-T"], fontsize=FS-2)
        ax[i].set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
plt.savefig("H3-11_unbiased_observers%.0f.pdf"%nc)
plt.tight_layout()
plt.show()
for i in range(nc):
    plt.errorbar(D, MM[i], yerr=MS[i], uplims=True, lolims=True)
plt.xlabel("Dosis (Gy)",fontsize=FS)
plt.ylabel("Mitotisk indeks (%)",fontsize=FS)
plt.tick_params(axis='both', which='major', labelsize=FS)
plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
plt.legend(["T47D", "T47D-P", "T47D-T"],fontsize=FS-2)

plt.tight_layout()
plt.savefig("H3-tgf-b3_%.0f.pdf"%nc)
#
plt.show()
