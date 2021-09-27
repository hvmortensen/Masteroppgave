import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'cm'
mpl.rcParams['font.sans-serif'] = 'Times New Roman'
mpl.rcParams['font.family'] = 'Times New Roman'

ku = 1
muen_over_rho = 1.075
pu = 1.02
kTP = 1.018
NK = 43.77  # mGy/nC
K = NK*ku*muen_over_rho*pu*kTP
dNK = 0.39
kTP = (273.15 + 37.1)/(273.15 + 24)

def multi_error(Z, a, da, b, db):
    return Z*np.sqrt(( (da/a)**2 + (db/b)**2))


def multi_error3(Z, a, da, b, db, c, dc):
    return Z*np.sqrt(( (da/a)**2 + (db/b)**2 + (dc/c)**2))


def add_error(da, db):
    return np.sqrt(da**2 + db**2)


### A = SSD50
A18 = np.mean([2.12, 2.11, 2.04, 2.10, 2.12, 2.07, 2.10, 2.11, 2.11, 2.12])
B18 = np.mean([2.10, 2.00, 2.13, 2.06, 1.99, 2.12, 2.06, 2.11, 2.07, 2.13])
C18 = np.mean([2.06, 2.04, 2.01, 2.06, 2.08, 2.07, 2.01, 1.98, 1.97, 2.11])
D18 = np.mean([2.02, 2.03, 2.09, 2.06, 2.08, 2.05, 1.99, 2.00, 2.07, 2.03])

A33 = np.mean([4.14, 4.14, 4.24, 4.20, 4.15, 4.25, 4.23, 4.20, 4.16, 4.17])
B33 = np.mean([4.25, 4.22, 4.22, 4.13, 4.18, 4.16, 4.11, 4.18, 4.22, 4.13])
C33 = np.mean([4.14, 4.05, 4.07, 4.15, 4.18, 4.07, 4.16, 4.09, 4.08, 4.19])
D33 = np.mean([4.12, 4.18, 4.12, 4.21, 4.10, 4.20, 4.20, 4.10, 4.11, 4.16])

A47 = np.mean([6.11, 6.15, 6.13, 6.22, 6.21, 6.11, 6.12, 6.17, 6.15, 6.09])
B47 = np.mean([6.24, 6.09, 6.14, 6.11, 6.12, 6.16, 6.16, 6.08, 6.14, 6.20])
C47 = np.mean([6.16, 6.10, 6.11, 6.12, 6.07, 6.16, 6.07, 6.11, 6.03, 6.14])
D47 = np.mean([6.12, 6.09, 6.07, 6.11, 6.03, 6.12, 6.08, 6.03, 6.12, 6.05])

A76 = np.mean([10.32, 10.16, 10.28, 10.21, 10.18, 10.19, 10.26, 10.22, 10.20, 10.19])
B76 = np.mean([10.23, 10.15, 10.23, 10.23, 10.18, 10.19, 10.14, 10.14, 10.18, 10.25])
C76 = np.mean([10.16, 10.16, 10.14, 10.23, 10.07, 10.13, 10.06, 10.13, 10.14, 10.14])
D76 = np.mean([10.21, 10.22, 10.11, 10.15, 10.16, 10.24, 10.18, 10.11, 10.22, 10.16])


m01 = np.mean([A18, B18, C18, D18])
m02 = np.mean([A33, B33, C33, D33])
m03 = np.mean([A47, B47, C47, D47])
m05 = np.mean([A76, B76, C76, D76])
s01 = np.std([A18, B18, C18, D18])/np.sqrt(4)
s02 = np.std([A33, B33, C33, D33])/np.sqrt(4)
s03 = np.std([A47, B47, C47, D47])/np.sqrt(4)
s05 = np.std([A76, B76, C76, D76])/np.sqrt(4)
print(s01)


### B = SSD40
A13 = np.mean([2.20, 2.04, 2.00, 1.98, 2.16])
B13 = np.mean([2.17, 1.99, 2.04, 2.11, 2.05])
C13 = np.mean([2.14, 1.99, 1.94, 2.05, 2.05])
D13 = np.mean([2.22, 2.09, 1.96, 2.05, 1.95])

A22 = np.mean([4.02, 4.06, 3.97, 3.96])
B22 = np.mean([4.11, 4.03, 3.98, 3.94])
C22 = np.mean([4.04])*kTP
D22 = np.mean([3.98])*kTP

A32 = np.mean([6.32, 6.15, 6.23])
B32 = np.mean([6.25, 6.13, 6.05])
C32 = np.mean([6.10, 6.02])*kTP
D32 = np.mean([5.92])*kTP


A51 = np.mean([10.30])
B51 = np.mean([10.17])
C51 = np.mean([9.98, 9.99])*kTP
D51 = np.mean([9.83])*kTP

A98 = np.mean([20.68])
B98 = np.mean([20.37])
C98 = np.mean([19.85])*kTP
D98 = np.mean([19.85])*kTP

n01 = np.mean([A13, B13, C13, D13])
n02 = np.mean([A22, B22, C22, D22])
n03 = np.mean([A32, B32, C32, D32])
n05 = np.mean([A51, B51, C51, D51])
n10 = np.mean([A98, B98, C98, D98])
r01 = np.std([A13, B13, C13, D13])/np.sqrt(4)
r02 = np.std([A22, B22, C22, D22])/np.sqrt(4)
r03 = np.std([A32, B32, C32, D32])/np.sqrt(4)
r05 = np.std([A51, B51, C51, D51])/np.sqrt(4)
r10 = np.std([A51, B51, C51, D51])/np.sqrt(4)


###SSD40
m40 = np.array([n01, n02, n03, n05, n10])
s40 = np.array([r01, r02, r03, r05, r10])
### SSD50
m50 = np.array([m01, m02, m03, m05])
s50 = np.array([s01, s02, s03, s05])



t40 = np.array([13, 22, 32, 51, 98])
t50 = np.array([18, 33, 47, 76])

dMu = 0.02
dD40 = np.zeros(5)
D40 = np.zeros(5)
for i in range(5):
    D40[i] = m40[i]*K
    dD40[i] = multi_error(D40[i], m40[i], add_error(s40[i], dMu), NK, dNK)
    print("Dosis ± Fejl for tid %.1f s = %.1f mGy ± %.1f %%" %(t40[i], D40[i], (dD40[i]/D40[i])*100) )

dD50 = np.zeros(4)
D50 = np.zeros(4)
for i in range(4):
    D50[i] = m50[i]*K
    dD50[i] = multi_error(D50[i], m50[i], add_error(s50[i], dMu), NK, dNK)
    print("Dosis ± Fejl for tid %.1f s = %.1f mGy ± %.1f %%" %(t50[i], D50[i], (dD50[i]/D50[i])*100) )


bt1 = np.array([3.58, 3.27, 4.14])
bt2 = np.array([3.34, 3.30, 2.20])
bt = np.mean([bt1, bt2])
dbt = np.std([bt1, bt2])/np.sqrt(2)
# nA = 0.219
dnA = 0.002
nA = 0.141
# t = 13 - bt
t = 18 - bt
dt = dbt
D = nA*K*t
dD = multi_error3(D, NK,dNK, nA,dnA, t, dt)
print(D, "±", dD/D*100)

# print("Fejlen i målingen = ", DMu, "nC")
### SSD40




# drt50 = np.array([6.9, 6.8, 6.8, 6.8])
# drs50 = np.array([])
# drt40 = np.array([])
# drs40 = np.array([10.3, 10.5, 10.5, 10.5, 10.6])
# ### Fejlen i dosen 0.1 Gy - 1.0 Gy
# dD = multi_error(D, Mu, DMu, NK, dNK)
# dDsem = add_error(SEM, dD)
# print("Dosen = ", D, "mGy ±", dDsem/D*100,"%")
# # print ("Fejlen i dosen = ", dDsem, "mGy")
# ### Ingen usikkerheder fra Pantak PMC1000 om eksponeringstid
# print("Build-up-tiden = ", bt, " ± ", dbt, "s")
# # print("Fejlen i build-up-tiden er = ", dbt, "s")
#
#
#
# drmean = np.mean(dr)
# drsem = np.std(dr)/np.sqrt(len(dr))
# print("Gennemsnitsdoseraten ± SEM =", drmean, "±", drsem)
#
#
# print("Doseraten for tid ", T, "s = ", Dr)
# print("Fejlen i den tekniske doserate = ", multi_error(Dr, Mu, DMu, NK, dNK))
# print("Fejlen i den sande doserate = ", multi_error3(Dr, Mu, DMu, NK ,dNK,bt, dbt))
# print("Fejlen i den sande doserate = ", multi_error(Dr, D, dDsem ,bt, dbt))
# print()
#




















# ddm = 0.002*m   # MAX4000 Repeatability
# dddm = 0.0006*m # MAX4000 Liniarity
