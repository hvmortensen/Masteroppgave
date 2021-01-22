import numpy as np
import matplotlib.pyplot as plt

# Pladen er et grid med 5 punkter i x-retning
# og 6 punkter i y-retning.
# 5 målinger er gjort i hvert punkt for at udregne
# gennemsnit med standardafvig.
# Målingerne gives i nC (nano Coulomb), dvs antal
# ladninger, da ioniseringskammeret "tæller"
# antallet af ioniserede luftmolekyler, dvs
# luftmolekyler som mister en elektron og dermed
# bliver ladet.


#omregningsfaktor K for dosimeteret i mG/nC (milliGray pr nanoCoulomb)
ku = 1
muen_over_rho = 1.075
pu = 1.02
kTP = 1.018
NK = 43.77
K = NK*ku*muen_over_rho*pu*kTP

a11 = np.array([1.54, 1.55, 1.54, 1.62, 1.58])
a21 = np.array([1.74, 1.65, 1.69, 1.68, 1.65])
a31 = np.array([1.67, 1.64, 1.74, 1.68, 1.72])
a41 = np.array([1.69, 1.61, 1.69, 1.69, 1.63])
a51 = np.array([1.48, 1.49, 1.45, 1.51, 1.51])

a12 = np.array([1.77, 1.74, 1.75, 1.72, 1.67])
a22 = np.array([1.83, 1.79, 1.87, 1.75, 1.86])
a32 = np.array([1.89, 1.82, 1.82, 1.87, 1.80])
a42 = np.array([1.79, 1.71, 1.76, 1.81, 1.79])
a52 = np.array([1.65, 1.59, 1.65, 1.61, 1.59])

a13 = np.array([1.83, 1.73, 1.80, 1.78, 1.79])
a23 = np.array([1.95, 1.85, 1.89, 1.90, 1.92])
a33 = np.array([1.93, 1.93, 1.88, 1.88, 1.92])
a43 = np.array([1.86, 1.86, 1.86, 1.89, 1.85])
a53 = np.array([1.78, 1.74, 1.78, 1.70, 1.69])

a14 = np.array([1.83, 1.84, 1.86, 1.83, 1.86])
a24 = np.array([1.93, 1.90, 1.91, 1.94, 1.90])
a34 = np.array([1.97, 1.90, 1.88, 1.94, 1.90])
a44 = np.array([1.91, 1.84, 1.83, 1.91, 1.84])
a54 = np.array([1.74, 1.80, 1.77, 1.77, 1.77])

a15 = np.array([1.86, 1.76, 1.81, 1.81, 1.79])
a25 = np.array([1.91, 1.90, 1.88, 1.90, 1.88])
a35 = np.array([1.97, 1.95, 1.92, 1.89, 1.89])
a45 = np.array([1.80, 1.86, 1.91, 1.80, 1.85])
a55 = np.array([1.76, 1.70, 1.73, 1.74, 1.70])

a16 = np.array([1.76, 1.68, 1.72, 1.73, 1.74])
a26 = np.array([1.76, 1.78, 1.81, 1.74, 1.83])
a36 = np.array([1.83, 1.85, 1.83, 1.81, 1.81])
a46 = np.array([1.77, 1.77, 1.77, 1.75, 1.79])
a56 = np.array([1.66, 1.67, 1.64, 1.59, 1.64])

#Gennemsnit (average) udregnes her
avg11 = np.mean(a11); avg21 = np.mean(a21); avg31 = np.mean(a31); avg41 = np.mean(a41); avg51 = np.mean(a51)
avg12 = np.mean(a12); avg22 = np.mean(a22); avg32 = np.mean(a32); avg42 = np.mean(a42); avg52 = np.mean(a52)
avg13 = np.mean(a13); avg23 = np.mean(a23); avg33 = np.mean(a33); avg43 = np.mean(a43); avg53 = np.mean(a53)
avg14 = np.mean(a14); avg24 = np.mean(a24); avg34 = np.mean(a34); avg44 = np.mean(a44); avg54 = np.mean(a54)
avg15 = np.mean(a15); avg25 = np.mean(a25); avg35 = np.mean(a35); avg45 = np.mean(a45); avg55 = np.mean(a55)
avg16 = np.mean(a16); avg26 = np.mean(a26); avg36 = np.mean(a36); avg46 = np.mean(a46); avg56 = np.mean(a56)

#Standardafvig (standard deviation) udregnes her
std11 = np.std(a11); std21 = np.std(a21); std31 = np.std(a31); std41 = np.std(a41); std51 = np.std(a51)
std12 = np.std(a12); std22 = np.std(a22); std32 = np.std(a32); std42 = np.std(a42); std52 = np.std(a52)
std13 = np.std(a13); std23 = np.std(a23); std33 = np.std(a33); std43 = np.std(a43); std53 = np.std(a53)
std14 = np.std(a14); std24 = np.std(a24); std34 = np.std(a34); std44 = np.std(a44); std54 = np.std(a54)
std15 = np.std(a15); std25 = np.std(a25); std35 = np.std(a35); std45 = np.std(a45); std55 = np.std(a55)
std16 = np.std(a16); std26 = np.std(a26); std36 = np.std(a36); std46 = np.std(a46); std56 = np.std(a56)

#Dose udregnes her
D11 = avg11*K; D21 = avg21*K; D31 = avg31*K; D41 = avg41*K; D51 = avg51*K
D12 = avg12*K; D22 = avg22*K; D32 = avg32*K; D42 = avg42*K; D52 = avg52*K
D13 = avg13*K; D23 = avg23*K; D33 = avg33*K; D43 = avg43*K; D53 = avg53*K
D14 = avg14*K; D24 = avg24*K; D34 = avg34*K; D44 = avg44*K; D54 = avg54*K
D15 = avg15*K; D25 = avg25*K; D35 = avg35*K; D45 = avg45*K; D55 = avg55*K
D16 = avg16*K; D26 = avg26*K; D36 = avg36*K; D46 = avg46*K; D56 = avg56*K
#
print ()
print ('I feltet (1,1) er CHRG =', "%.2f" % avg11, '±', "%.2f" % std11, 'nC, og Dose =', "%.2f" % D11, 'mGy')
print ('I feltet (2,1) er CHRG =', "%.2f" % avg21, '±', "%.2f" % std21, 'nC, og Dose =', "%.2f" % D21, 'mGy')
print ('I feltet (3,1) er CHRG =', "%.2f" % avg31, '±', "%.2f" % std31, 'nC, og Dose =', "%.2f" % D31, 'mGy')
print ('I feltet (4,1) er CHRG =', "%.2f" % avg41, '±', "%.2f" % std41, 'nC, og Dose =', "%.2f" % D41, 'mGy')
print ('I feltet (5,1) er CHRG =', "%.2f" % avg51, '±', "%.2f" % std51, 'nC, og Dose =', "%.2f" % D51, 'mGy')
print()
print ('I feltet (1,2) er CHRG =', "%.2f" % avg12, '±', "%.2f" % std12, 'nC, og Dose =', "%.2f" % D12, 'mGy')
print ('I feltet (2,2) er CHRG =', "%.2f" % avg22, '±', "%.2f" % std22, 'nC, og Dose =', "%.2f" % D22, 'mGy')
print ('I feltet (3,2) er CHRG =', "%.2f" % avg32, '±', "%.2f" % std32, 'nC, og Dose =', "%.2f" % D32, 'mGy')
print ('I feltet (4,2) er CHRG =', "%.2f" % avg42, '±', "%.2f" % std42, 'nC, og Dose =', "%.2f" % D42, 'mGy')
print ('I feltet (5,2) er CHRG =', "%.2f" % avg52, '±', "%.2f" % std52, 'nC, og Dose =', "%.2f" % D52, 'mGy')
print ()
print ('I feltet (1,3) er CHRG =', "%.2f" % avg13, '±', "%.2f" % std13, 'nC, og Dose =', "%.2f" % D13, 'mGy')
print ('I feltet (2,3) er CHRG =', "%.2f" % avg23, '±', "%.2f" % std23, 'nC, og Dose =', "%.2f" % D23, 'mGy')
print ('I feltet (3,3) er CHRG =', "%.2f" % avg33, '±', "%.2f" % std33, 'nC, og Dose =', "%.2f" % D33, 'mGy')
print ('I feltet (4,3) er CHRG =', "%.2f" % avg43, '±', "%.2f" % std43, 'nC, og Dose =', "%.2f" % D43, 'mGy')
print ('I feltet (5,3) er CHRG =', "%.2f" % avg53, '±', "%.2f" % std53, 'nC, og Dose =', "%.2f" % D53, 'mGy')
print ()
print ('I feltet (1,4) er CHRG =', "%.2f" % avg14, '±', "%.2f" % std14, 'nC, og Dose =', "%.2f" % D14, 'mGy')
print ('I feltet (2,4) er CHRG =', "%.2f" % avg24, '±', "%.2f" % std24, 'nC, og Dose =', "%.2f" % D24, 'mGy')
print ('I feltet (3,4) er CHRG =', "%.2f" % avg34, '±', "%.2f" % std34, 'nC, og Dose =', "%.2f" % D34, 'mGy')
print ('I feltet (4,4) er CHRG =', "%.2f" % avg44, '±', "%.2f" % std44, 'nC, og Dose =', "%.2f" % D44, 'mGy')
print ('I feltet (5,4) er CHRG =', "%.2f" % avg54, '±', "%.2f" % std54, 'nC, og Dose =', "%.2f" % D54, 'mGy')
print ()
print ('I feltet (1,5) er CHRG =', "%.2f" % avg15, '±', "%.2f" % std15, 'nC, og Dose =', "%.2f" % D15, 'mGy')
print ('I feltet (2,5) er CHRG =', "%.2f" % avg25, '±', "%.2f" % std25, 'nC, og Dose =', "%.2f" % D25, 'mGy')
print ('I feltet (3,5) er CHRG =', "%.2f" % avg35, '±', "%.2f" % std35, 'nC, og Dose =', "%.2f" % D35, 'mGy')
print ('I feltet (4,5) er CHRG =', "%.2f" % avg45, '±', "%.2f" % std45, 'nC, og Dose =', "%.2f" % D45, 'mGy')
print ('I feltet (5,5) er CHRG =', "%.2f" % avg55, '±', "%.2f" % std55, 'nC, og Dose =', "%.2f" % D55, 'mGy')
print ()
print ('I feltet (1,6) er CHRG =', "%.2f" % avg16, '±', "%.2f" % std16, 'nC, og Dose =', "%.2f" % D16, 'mGy')
print ('I feltet (2,6) er CHRG =', "%.2f" % avg26, '±', "%.2f" % std26, 'nC, og Dose =', "%.2f" % D26, 'mGy')
print ('I feltet (3,6) er CHRG =', "%.2f" % avg36, '±', "%.2f" % std36, 'nC, og Dose =', "%.2f" % D36, 'mGy')
print ('I feltet (4,6) er CHRG =', "%.2f" % avg46, '±', "%.2f" % std46, 'nC, og Dose =', "%.2f" % D46, 'mGy')
print ('I feltet (5,6) er CHRG =', "%.2f" % avg56, '±', "%.2f" % std56, 'nC, og Dose =', "%.2f" % D56, 'mGy')
