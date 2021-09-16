import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'cm'
mpl.rcParams['font.sans-serif'] = 'Times New Roman'
mpl.rcParams['font.family'] = 'Times New Roman'

D = np.array([0, 0.1, 0.2, 0.3, 0.5, 1.0])


H9_1 = 1.4
H9_05 = 2.8
H9_03 = 2.1
H9_02 = 2.6
H9_01 = 3.0
H9_k = 3.3
H9_1p = 0.6
H9_05p = 2.6
H9_03p = 1.7
H9_02p = 1.9
H9_01p = 1.5
H9_kp = 3.7

H10_1 = np.mean([1.2, 1.4, 1.1, 1.0, 0.9])
H10_05 = np.mean([1.9, 2.1, 1.9, 1.8, 1.6])
H10_03 = np.mean([2.8, 2.7, 2.6, 2.6, 2.4])
H10_02 = np.mean([2.5, 2.4, 2.4, 2.4, 2.2])
H10_01 = np.mean([2.8, 2.7, 2.6, 2.6, 2.5])
H10_k = np.mean([2.8, 3.0, 2.7, 2.6, 2.6])
H10_1p = np.mean([2.0, 1.9, 1.9, 1.8, 1.4])
H10_05p = np.mean([1.4, 1.3, 1.1, 1.3, 1.5])
H10_03p = np.mean([2.5, 2.6, 2.3, 2.4, 2.5])
H10_02p = np.mean([2.5, 2.2, 2.3, 2.3, 2.3])
H10_01p = np.mean([2.4, 2.5, 2.7, 2.6, 2.8])
H10_kp = np.mean([3.8, 3.7, 3.1, 3.6, 3.3])



H11_1 = np.mean([1.9, 2.5, 2.5, 2.5])
H11_05 = np.mean([1.8, 1.9, 1.9, 2.2])
H11_03 = np.mean([1.8, 2.0, 1.8, 1.7])
H11_02 = np.mean([2.7, 2.6, 2.9, 2.7])
H11_01 = np.mean([2.6, 2.8, 3.2, 2.8])
H11_k = np.mean([2.8, 2.6, 2.7, 2.6])
H11_1p = np.mean([1.1, 1.6, 1.2, 1.1])
H11_05p = np.mean([2.0, 2.0, 1.9, 1.8])
H11_03p = np.mean([2.2, 2.2, 2.4, 2.2])
H11_02p = np.mean([3.2, 2.8, 3.3, 2.7])
H11_01p = np.mean([2.3, 2.0, 3.0, 2.0])
H11_kp = np.mean([3.4, 3.2, 3.6, 2.9])


H15_1 = 0.5#np.mean([0.5, 0.9, 0.8, 0.8, 1.1])
H15_05 = 0.7#np.mean([0.7, 0.8, 0.8, 0.9, 0.9])
H15_03 = 0.8#np.mean([0.8, 0.7, 0.8, 0.9, 0.9])
H15_02 = 2.1#np.mean([1.6, 1.5, 1.9, 1.7, 1.8])
H15_01 = 1.6#np.mean([1.3, 1.2, 1.4, 1.3, 1.1])
H15_k = 1.3#np.mean([1.2, 1.0, 1.2, 1.2, 1.0])
H15_1p = 0.8#np.mean([0.8, 0.7, 0.8, 0.8, 1.1])
H15_05p = 1.5#np.mean([1.6, 1.4, 1.6, 1.7, 1.4])
H15_03p = 0.6#np.mean([1.0, 0.9, 0.9, 1.1, 1.1])
H15_02p = 1.6#np.mean([1.5, 1.0, 1.4, 1.6, 1.4])
H15_01p = 2.1#np.mean([2.0, 2.0, 1.8, 2.5, 2.5])
H15_kp = 1.9#np.mean([1.7, 1.6, 1.6, 1.8, 1.5])


MI_H9 = np.array([H9_k, H9_01, H9_02, H9_03, H9_05, H9_1])/H9_k*100
MI_H9p = np.array([H9_kp, H9_01p, H9_02p, H9_03p, H9_05p, H9_1p])/H9_kp*100
MI_H10 = np.array([H10_k, H10_01, H10_02, H10_03, H10_05, H10_1])/H10_k*100
MI_H10p = np.array([H10_kp, H10_01p, H10_02p, H10_03p, H10_05p, H10_1p])/H10_kp*100
MI_H11 = np.array([H11_k, H11_01, H11_02, H11_03, H11_05, H11_1])/H11_k*100
MI_H11p = np.array([H11_kp, H11_01p, H11_02p, H11_03p, H11_05p, H11_1p])/H11_kp*100
MI_H15 = np.array([H15_k, H15_01, H15_02, H15_03, H15_05, H15_1])/H15_k*100
MI_H15p = np.array([H15_kp, H15_01p, H15_02p, H15_03p, H15_05p, H15_1p])/H15_kp*100





MI_avg = np.zeros(len(D))
MIp_avg = np.zeros(len(D))
MI_std = np.zeros(len(D))
MIp_std = np.zeros(len(D))
for i in range(len(D)):

    MIall = np.array([MI_H9[i], MI_H10[i], MI_H11[i]])
    MIpall = np.array([MI_H9[i], MI_H10p[i], MI_H11p[i]])

    MI_avg[i] = np.mean(MIall)
    MIp_avg[i] = np.mean(MIpall)
    MI_std[i] = np.std(MIall)
    MIp_std[i] = np.std(MIpall)

FS = 17


fig, ax = plt.subplots(2,3, figsize=(14,8.5),sharey="all")
ax[0,0].plot(D, MI_H9, label="H9, normal")
ax[0,0].plot(D, MI_H9p, label="H9, primet")
ax[0,1].plot(D, MI_H10, label="H10, normal")
ax[0,1].plot(D, MI_H10p,label="H10, primet")
ax[0,2].plot(D, MI_H11, label="H11, normal")
ax[0,2].plot(D, MI_H11p,label="H11, primet")


for i in range(2):
    for j in range(3):
        ax[i,j].legend(fontsize=FS-2)
        ax[i,j].set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax[i,j].set_xlabel("Dosis (Gy)",fontsize=FS)
        ax[i,j].tick_params(axis='both', which='major', labelsize=FS)
        ax[i,0].set_ylabel("Mitotisk index (%)",fontsize=FS)
plt.tight_layout()
plt.savefig("mitoticindex.pdf")
plt.show()

#
# figt, axt = plt.subplots(1,p, figsize=(14,4.5))#,sharey="all")
# axt[0].plot(D, MI_H13, label="H10, normal")
# axt[0].plot(D, MI_H13t, label="H10, TGF-B3")
# # ax[1].plot(D, MI_H11, label="H11, normal")
# # ax[1].plot(D, MI_H11p,label="H11, primet")
# # ax[2].plot(D, MI_H12, label="H12, normal")
# # ax[2].plot(D, MI_H12p, label="H12, primet")
#
# for i in range(p):
#
#     axt[i].legend(fontsize=FS-2)
#     axt[i].set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
#     axt[i].set_xlabel("Dosis (Gy)",fontsize=FS)
#     # ax[i].set_ylabel("Mitotisk index (%)",fontsize=FS)
#     axt[i].tick_params(axis='both', which='major', labelsize=FS)
#
# axt[0].set_ylabel("Mitotisk index (%)",fontsize=FS)
# plt.tight_layout()
# plt.savefig("mitoticindex.pdf")
# plt.show()



plt.errorbar(D, MI_avg, yerr=MI_std, uplims=True, lolims=True,label="normal")
plt.errorbar(D, MIp_avg, yerr=MIp_std,uplims=True, lolims=True, label="primet")
plt.xlabel("Dosis (Gy)",fontsize=FS)
plt.ylabel("Mitotisk indeks (%)",fontsize=FS)
plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
plt.tick_params(axis='both', which='major', labelsize=FS)
plt.legend(fontsize=FS-2)
plt.tight_layout()
plt.savefig("mitoticindex_avg.pdf")
plt.show()
