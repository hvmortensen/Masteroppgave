import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'cm'
mpl.rcParams['font.sans-serif'] = 'Times New Roman'
mpl.rcParams['font.family'] = 'Times New Roman'

number_of_observers = 4
no = number_of_observers


# D = np.array([0.0, 0.1, 0.2, 0.3, 0.5])
D = np.array([0.0, 0.1, 0.2, 0.3, 0.5, 1.0])
dl = len(D)

if dl == 6:
    number_of_celltypes = 2
    nc = number_of_celltypes
    H = np.loadtxt("H3-11_henrik.txt")
    N = np.loadtxt("H3-11_nina.txt")
    I = np.loadtxt("H3-11_ingunn.txt")
    J = np.loadtxt("H3-11_jenny.txt")
elif dl == 5:
    number_of_celltypes = 3
    nc = number_of_celltypes
    H = np.loadtxt("H3-18_henrik.txt")
    N = np.loadtxt("H3-18_nina.txt")
    I = np.loadtxt("H3-18_ingunn.txt")
    J = np.loadtxt("H3-18_jenny.txt")

M = np.zeros((no, nc, dl))
S = np.zeros((no, nc, dl))

for i in range(dl):
    for j in range(nc):
        M[0,j,i] = np.mean(H[i + j*dl])/np.mean(H[j*dl])
        M[1,j,i] = np.mean(N[i + j*dl])/np.mean(N[j*dl])
        M[2,j,i] = np.mean(I[i + j*dl])/np.mean(I[j*dl])
        M[3,j,i] = np.mean(J[i + j*dl])/np.mean(J[j*dl])
        S[0,j,i] = np.std(H[i + j*dl])/np.mean(H[j*dl])
        S[1,j,i] = np.std(N[i + j*dl])/np.mean(N[j*dl])
        S[2,j,i] = np.std(I[i + j*dl])/np.mean(I[j*dl])
        S[3,j,i] = np.std(J[i + j*dl])/np.mean(J[j*dl])

MM = np.zeros((nc,dl))
MS = np.zeros((nc,dl))
for i in range(nc):
    for j in range(dl):
        MM[i,j] = np.mean([M[0,i,j], M[1,i,j], M[2,i,j], M[3,i,j]])
        MS[i,j] = np.std([M[0,i,j], M[1,i,j], M[2,i,j], M[3,i,j]])/np.sqrt(no)

FS = 17
fig, ax = plt.subplots(1,4, figsize=(14,4.5),sharey="all")
for i in range(no):
    for j in range(nc):
        ax[i].errorbar(D, M[i,j], yerr=S[i,j], uplims=True, lolims=True)
        ax[0].text(0.05, 0.95, "Henrik", transform=ax[0].transAxes)
        ax[1].text(0.05, 0.95, "Nina", transform=ax[1].transAxes)
        ax[2].text(0.20, 0.95, "Ingunn", transform=ax[2].transAxes)
        ax[3].text(0.05, 0.95, "Jenny", transform=ax[3].transAxes)
        ax[i].set_xlabel("Dosis (Gy)",fontsize=FS)
        ax[0].set_ylabel("Mitotisk index (%)",fontsize=FS)
        ax[i].tick_params(axis='both', which='major', labelsize=FS)
        if dl == 6:
            ax[no-1].legend(["T47D", "T47D-P"], fontsize=FS-2)
            ax[i].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        elif dl == 5:
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
if dl == 6:
    plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.legend(["T47D", "T47D-P"],fontsize=FS-2)
elif dl == 5:
    plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    plt.legend(["T47D", "T47D-P", "T47D-T"],fontsize=FS-2)

plt.tight_layout()
plt.savefig("H3-11_mean_unbiased_%.0f.pdf"%nc)
#
plt.show()
