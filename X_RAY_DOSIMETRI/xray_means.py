import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cycler
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox, AnchoredOffsetbox
import matplotlib.image as mpimg
import matplotlib as mpl

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'cm'
mpl.rcParams['font.sans-serif'] = 'Times New Roman'
mpl.rcParams['font.family'] = 'Times New Roman'


#### LILLE PLOT MED KALIBRERINGSFAKTOR FOR FORSKELLIGE EKSPONERINGSTIDER ####
scope = np.array([12, 13, 14, 20])

avg0 = np.mean(NormArray[0])
std0 = np.std(NormArray[0])
sem0 = np.std(NormArray[0])/np.sqrt(len(NormArray[0]))
# print(avg0)
# print(std0)
# print(sem0)
first_meas_norms = np.array([1.0215465117895153, 1.0204540638976112, 1.0120538668226677, 1.0087396484416213 ])
first_meas_stds = np.array([0.0353466377428203, 0.02763525154902737, 0.026603320855124565,0.016202058486188126])
first_meas_sems = np.array([0.006453383607902055, 0.0029130112868874284, 0.002804236240881732, 0.0017078469199871408 ])


#### KALIBREINGSFAKTOR MED RENSEDE NORMALISERINGER SÅ FØRSTEINDEKSEN IKEN MEDGÅR I GENNEMSNITTET ####
avg0_wi0 = np.mean(NormArray_wi0[0])
std0_wi0 = np.std(NormArray_wi0[0])
sem0_wi0 = np.std(NormArray_wi0[0])/np.sqrt(len(NormArray_wi0[0]))
# print(avg0_wi0)
# print(std0_wi0)
# print(sem0_wi0)
first_meas_norms_wi0 = np.array([1.0241531756700784, 1.0228734481098294, 1.0153356249882308, 1.0110310657488206])
first_meas_stds_wi0 = np.array([0.039411262667292044, 0.030849849067484964, 0.033392861579865804, 0.020344157007350143])
first_meas_sems_wi0 = np.array([0.007195479194212357,0.003251859617522467, 0.0035199166727701535, 0.0021444624406433783])



#### GENNEMSNITLIG STANDARDAFVIG I DET HOMOGENE OMRÅDE AF EKSPONERINGSREGIONEN ####
P_Hom_mean = np.mean(PS_Matrix[:,2:-1,1:-1])
P_Hom_std = np.std(PS_Matrix[:,2:-1,1:-1])
P_Hom_sem = np.std(PS_Matrix[:,2:-1,1:-1])/np.sqrt(PS_Matrix[:,2:-1,1:-1].shape[0]*PS_Matrix[:,2:-1,1:-1].shape[1])
# print((PS_Matrix[:,2:-1,1:-1]))
# print( PS_Matrix[:,2:-1,1:-1].shape[0]*PS_Matrix[:,2:-1,1:-1].shape[1])
print("P_Hom_mean =", P_Hom_mean)
print("P_Hom_std =", P_Hom_std)
print("P_Hom_sem =", P_Hom_sem)
meanstd_hom = np.array([3.0609518709544905, 2.9058740060196473, 2.5327468220836034, 1.6194550102161929])
stdstd_hom = np.array([0.508958148510818, 0.618839073735935, 0.6038044325544453, 0.5369453932178103])
semstd_hom = np.array([0.29384712404897434, 0.20627969124531167, 0.2012681441848151, 0.17898179773927012])

PSEM_Hom_mean = np.mean(PSEM_Matrix[2:-1,1:-1])
PSEM_Hom_std = np.std(PSEM_Matrix[2:-1,1:-1])
PSEM_Hom_sem = np.std(PSEM_Matrix[2:-1,1:-1])/np.sqrt(PS_Matrix[:,2:-1,1:-1].shape[0]*PS_Matrix[:,2:-1,1:-1].shape[1])

print("PSEM_Hom_mean =", PSEM_Hom_mean)
print("PSEM_Hom_std =", PSEM_Hom_std)
print("PSEM_Hom_sem =", PSEM_Hom_sem)

scope_sem = np.array([13,14,20])
meansem_hom = np.array([0.8510225461213561,1.3031104776606717,1.5848983129957606])
semsem_hom = np.array([0.053547249218584086,0.15708604238204218,0.07614570548795169])

PSE_Hom_mean = np.mean(PSE_Matrix[2:-1,1:-1])
PSE_Hom_std = np.std(PSE_Matrix[2:-1,1:-1])
PSE_Hom_sem = np.std(PSE_Matrix[2:-1,1:-1])/np.sqrt(PS_Matrix[:,2:-1,1:-1].shape[0]*PS_Matrix[:,2:-1,1:-1].shape[1])

print("PSE_Hom_mean =", PSE_Hom_mean)
print("PSE_Hom_std =", PSE_Hom_std)
print("PSE_Hom_sem =", PSE_Hom_sem)

scope_se = np.array([12, 13,14,20])
meanse_hom = np.array([0.9679579720369986,0.6064155326271928,0.9079906658732457,0.8353226395519263])
semse_hom = np.array([0.09292261958847979,0.015456749637387001,0.03778216140825923,0.029387005381193845])



titlee = "Kalibreringsfaktor $\\delta(t)$ for kort eksponering"
fige, axe = plt.subplots()
# axe.set_title(titlee, fontsize=FS)
axe.plot(scope,first_meas_norms, "r", label="$\\delta(t)$")
axe.fill_between(scope, first_meas_norms-first_meas_sems, first_meas_norms+first_meas_sems, alpha=0.3,label="SEM")
axe.set_xlabel("Eksponeringstid (s)",fontsize=FS)
axe.set_ylabel("Kalibreringsfaktor $\\delta(t)$",fontsize=FS)
axe.legend(fontsize = FS)
# axe.set_yticks([1.006, 1.008, 1.010, 1.012, 1.014, 1.016, 1.018, 1.020, 1.022, 1.024, 1.026, 1.028])
plt.xticks(fontsize=FS)
plt.yticks(fontsize=FS)
plt.tight_layout()
# plt.tight_layout(rect=titlecorrection)
plt.savefig("Kalib_faktor_over_tid.pdf")

# titlef = "Gennemsnitligt standardafvig i det homogene felt som funktion af eksponeringstid"
figf, axf = plt.subplots()
# axe.set_title(titlee, fontsize=FS)
im = mpimg.imread('gitter.png')
axf.imshow(im, extent=[11.5,14.65,1.5,2.26], aspect='auto')
axf.plot(scope,meanstd_hom, label="$\\langle$SD$\\rangle(t)$")
axf.fill_between(scope, meanstd_hom - semstd_hom, meanstd_hom + semstd_hom,alpha=0.3,label="SEM")
axf.set_xlabel("Eksponeringstid (s)",fontsize=FS)
axf.set_ylabel("Gnsn standardafvig $\\langle$SD$\\rangle(t)$ (%)",fontsize=FS)
axf.legend(fontsize = FS)
plt.xticks(fontsize=FS)
plt.yticks(fontsize=FS)
plt.tight_layout()
# plt.tight_layout(rect=titlecorrection)
plt.savefig("Gnsn_std_afvig.pdf")
#
titleg = "Kalibreringsfaktor $\\delta(t)$ for kort eksponering uden førstemåling"
figg, axg = plt.subplots()
# axg.set_title(titlee, fontsize=FS)
axg.plot(scope,first_meas_norms_wi0, "m", label="$\\delta(t)$ (uden indeks 1)")
axg.fill_between(scope, first_meas_norms_wi0-first_meas_sems_wi0, first_meas_norms_wi0+first_meas_sems_wi0,alpha=0.3,label="SEM")
axg.plot(scope,first_meas_norms, "r--", label="$\\delta(t)$")
axg.fill_between(scope, first_meas_norms-first_meas_sems, first_meas_norms+first_meas_sems,alpha=0.3,label="SEM")
axg.set_xlabel("Eksponeringstid (s)",fontsize=FS)
axg.set_ylabel("Kalibreringsfaktor $\\delta(t)$",fontsize=FS)
axg.legend(fontsize = FS)
# axg.set_yticks([1.006, 1.008, 1.010, 1.012, 1.014, 1.016, 1.018, 1.020, 1.022, 1.024, 1.026, 1.028])
plt.xticks(fontsize=FS)
plt.yticks(fontsize=FS)
plt.tight_layout()
# plt.tight_layout(rect=titlecorrection)
plt.savefig("Kalib_faktor_over_tid_renset.pdf")


# titlef = "Gennemsnitligt standardfejl i det homogene felt som funktion af eksponeringstid"
figh, axh = plt.subplots()
# axe.set_title(titlee, fontsize=FS)
im = mpimg.imread('gitter.png')
# axh.imshow(im, extent=[17.3,20.3,1.07,1.47], aspect='auto')
axh.imshow(im, extent=[12.6+5,15.2+5,0.8,1.13], aspect='auto')
axh.plot(scope_sem,meansem_hom, "g", label="$\\langle$SEM$\\rangle(t)$")

axh.fill_between(scope_sem, meansem_hom - semsem_hom, meansem_hom + semsem_hom,alpha=0.3,label="SEM")
axh.set_xlabel("Eksponeringstid (s)",fontsize=FS)
axh.set_ylabel("Gnsn standardfejl $\\langle$SEM$\\rangle(t)$ (%)",fontsize=FS)
axh.legend(fontsize = FS)
plt.xticks(fontsize=FS)
plt.yticks(fontsize=FS)
plt.tight_layout()
# plt.tight_layout(rect=titlecorrection)
plt.savefig("Gnsn_SEM.pdf")




#
# titleg = "Kalibreringsfaktor $\\delta(t)$ for kort eksponering (renset)"
# figg, axg = plt.subplots()
# # axg.set_title(titleg, fontsize=FS)
# axg.plot(scope,first_meas_norms_wi, "r", label="$\\delta(t)$")
# axg.fill_between(scope, first_meas_norms_wi-first_meas_sems_wi, first_meas_norms_wi+first_meas_sems_wi,alpha=0.3,label="SEM")
# axg.set_xlabel("Eksponeringstid (s)",fontsize=fs)
# axg.set_ylabel("Kalibreringsfaktor $\\delta(t)$",fontsize=fs)
# axg.legend(fontsize = fs)
# plt.xticks(fontsize=fs)
# plt.yticks(fontsize=fs)
# plt.tight_layout(rect=titlecorrection)
# plt.savefig("Kalib_faktor_over_tid_renset.pdf")
#



# titlef = "Gennemsnitlig standardafvig i % for kort eksponering"
# figf, axf = plt.subplots()
# # axf.set_title(titlef, fontsize=FS)
# axf.plot(scope,meanstd, "r", label="$\\langle$SD$\\rangle $")
# axf.fill_between(scope, meanstd-stdstd, meanstd+stdstd,alpha=0.3,label="SD($\\langle$SD$\\rangle $)")
# axf.set_xlabel("Eksponeringstid (s)", fontsize=fs)
# axf.set_ylabel("SD (%)", fontsize=fs)
# axf.legend(fontsize = fs)
# plt.xticks(fontsize=fs)
# plt.yticks(fontsize=fs)
# plt.tight_layout(rect=titlecorrection)
# plt.savefig("Standardafvigiprocent.pdf")

# plt.show()
