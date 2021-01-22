import numpy as np
import matplotlib.pyplot as plt


#omregningsfaktor for dosimeteret i mGy/nC (milliGray pr nanoCoulomb)
ku = 1
muen_over_rho = 1.075
pu = 1.02
kTP = 1.018
NK = 43.77
K = NK*ku*muen_over_rho*pu*kTP


# ## Målinger for t sekunder
# ## Plexiboxen blev skubbet lidt rundt for at finde det bedste sted:
# ## + 13 cm fra døren og - 13 cm fra højre kant
# ## Tape med sort streg markerer positionen og flaskespidserne peger i y-retning

# ## De bedste resultater NB! FORKERT OMREGNINGSFAKTOR
# ## For 0.1 Gy fandt vi 20 s til at give en gennemsnitlig dose på 0.105 Gy
# ## For 0.2 Gy fandt vi 36 s til at give en gennemsnitlig dose på 0.202 Gy
# ## For 0.3 Gy fandt vi 52 s til at give en gennemsnitlig dose på 0.302 Gy
# ## For 0.5 Gy fandt vi 85 s til at give en gennemsnitlig dose på 0.302 Gy

#
#
# t = 19
# a1 = np.array([2.20, 2.31, 2.15, 2.25, 2.22, 2.26, 2.21, 2.23, 2.29, 2.23])
# a2 = np.array([2.32, 2.32, 2.26, 2.25, 2.28, 2.30, 2.26, 2.27, 2.18, 2.27])
# a3 = np.array([2.21, 2.20, 2.26, 2.16, 2.15, 2.30, 2.28, 2.21, 2.23, 2.21])
# a4 = np.array([2.18, 2.32, 2.15, 2.18, 2.19, 2.18, 2.28, 2.19, 2.17, 2.22])
#

#
# t = 20
# a1 = np.array([2.35, 2.46, 2.45, 2.43, 2.39, 2.40, 2.45, 2.39, 2.36, 2.34])
# a2 = np.array([2.47, 2.40, 2.35, 2.39, 2.39, 2.36, 2.44, 2.39, 2.36, 2.43])
# a3 = np.array([2.38, 2.36, 2.32, 2.42, 2.38, 2.36, 2.39, 2.44, 2.34, 2.41])
# a4 = np.array([2.36, 2.34, 2.31, 2.41, 2.31, 2.39, 2.41, 2.41, 2.40, 2.41])

# t = 35
# a1 = np.array([4.57, 4.58, 4.47, 4.47, 4.46, 4.54, 4.46, 4.51, 4.44, 4.50])
# a2 = np.array([4.51, 4.51, 4.48, 4.49, 4.52, 4.49, 4.55, 4.48, 4.52, 4.55])
# a3 = np.array([4.55, 4.42, 4.48, 4.55, 4.55, 4.51, 4.42, 4.43, 4.44, 4.54])
# a4 = np.array([4.42, 4.52, 4.56, 4.45, 4.47, 4.44, 4.44, 4.50, 4.49, 4.51])

# t = 36
# a1 = np.array([4.68, 4.61, 4.64, 4.65, 4.69, 4.62, 4.71, 4.62, 4.66, 4.64])
# a2 = np.array([4.62, 4.60, 4.69, 4.63, 4.60, 4.60, 4.62, 4.66, 4.64, 4.62])
# a3 = np.array([4.60, 4.55, 4.59, 4.62, 4.57, 4.59, 4.60, 4.66, 4.56, 4.67])
# a4 = np.array([4.59, 4.56, 4.67, 4.64, 4.55, 4.57, 4.59, 4.66, 4.60, 4.55])



# t = 51
# a1 = np.array([6.71, 6.66, 6.79, 6.70, 6.76])
# a2 = np.array([6.78, 6.69, 6.71, 6.79, 6.71])
# a3 = np.array([6.71, 6.68, 6.67, 6.72, 6.74])
# a4 = np.array([6.70, 6.67, 6.66, 6.71, 6.65])

# t = 52
# a1 = np.array([6.98, 6.93, 6.92, 6.95, 6.99])
# a2 = np.array([6.96, 6.83, 6.89, 6.96, 6.89])
# a3 = np.array([6.91, 6.83, 6.91, 6.88, 6.88])
# a4 = np.array([6.84, 6.86, 6.81, 6.80, 6.88])

#
# t = 53
# a1 = np.array([7.14, 7.09, 7.12, 7.06, 6.99])
# a2 = np.array([7.09, 7.06, 6.99, 6.98, 7.03])
# a3 = np.array([7.06, 6.99, 6.99, 7.06, 7.03])
# a4 = np.array([7.02, 7.03, 7.07, 7.06, 6.93])
#


# t = 85
# a1 = np.array([11.50, 11.48, 11.59, 11.53, 11.53])
# a2 = np.array([11.56, 11.54, 11.56, 11.55, 11.45])
# a3 = np.array([11.40, 11.39, 11.47, 11.36, 11.39])
# a4 = np.array([11.50, 11.39, 11.44, 11.49, 11.45])
#


# #### SSD40 ####
# t = 14
# a1 = np.array([2.35, 2.19, 2.12, 2.17, 2.19])
# a2 = np.array([2.35, 2.25, 2.25, 2.27, 2.17])
# a3 = np.array([2.24, 2.15, 2.21, 2.17, 2.20])
# a4 = np.array([2.16, 2.21, 2.18, 2.13, 2.31])

# t = 15
# a1 = np.array([2.50, 2.43, 2.49, 2.44, 2.53])
# a2 = np.array([2.45, 2.38, 2.49, 2.56, 2.45])
# a3 = np.array([2.42, 2.31, 2.49, 2.42, 2.45])
# a4 = np.array([2.53, 2.34, 2.33, 2.33, 2.33])
#
# t = 10
# a1 = np.array([1.46, 1.54, 1.52, 1.53, 1.51])
# a2 = np.array([1.43, 1.46, 1.41, 1.42, 1.37])
# a3 = np.array([1.43, 1.31, 1.31])
# a4 = np.array([])
# # #
# t = 12
# a1 = np.array([])
# a2 = np.array([])
# a3 = np.array([1.82, 1.91, 1.72, 1.91, 1.81])
# a4 = np.array([])
#
#
# t = 13
# a1 = np.array([2.20, 2.04, 2.00, 1.98, 2.16])
# a2 = np.array([2.17, 1.99, 2.04, 2.11, 2.05])
# a3 = np.array([2.14, 1.99, 1.94, 2.05, 2.05])
# a4 = np.array([2.22, 2.09, 1.96, 2.05, 1.95])
# #

# #### SSD37_5 ####
# t = 12
# a1 = np.array([2.05, 2.04, 2.04, 1.95, 1.95])
# a2 = np.array([2.01, 2.07, 1.92, 2.00, 2.06])
# a3 = np.array([1.96, 2.05, 1.99, 2.05, 2.03])
# a4 = np.array([2.12, 2.00, 2.06, 1.90, 1.94])
# #
t = 13
a1 = np.array([])
a2 = np.array([])
a3 = np.array([])
a4 = np.array([2.17, 2.29, 2.24, 2.23, 2.10])
# #
# t = 12
# a1 = np.array([])
# a2 = np.array([])
# a3 = np.array([])
# a4 = np.array([])
# #

avg1 = np.mean(a1); avg2 = np.mean(a2); avg3 = np.mean(a3); avg4 = np.mean(a4)
std1 = np.std(a1); std2 = np.std(a2); std3 = np.std(a3); std4 = np.std(a4)
D1 = avg1*K; D2 = avg2*K; D3 = avg3*K; D4 = avg4*K
D1std = std1*K; D2std = std2*K; D3std = std3*K; D4std = std4*K

print()
print ()
print ('Eksponering =', t, 'sekunder')
print ()
print ('Rum 1 CHRG =', "%.2f" % avg1, '±', "%.2f" % std1, 'nC')
print ('Rum 1 Dose =', "%.2f" % D1, 'mGy')
print ()
print ('Rum 2 CHRG =', "%.2f" % avg2, '±', "%.2f" % std2, 'nC')
print ('Rum 2 Dose =', "%.2f" % D2, 'mGy')
print ()
print ('Rum 3 CHRG =', "%.2f" % avg3, '±', "%.2f" % std3, 'nC')
print ('Rum 3 Dose =', "%.2f" % D3, 'mGy')
print ()
print ('Rum 4 CHRG =', "%.2f" % avg4, '±', "%.2f" % std4, 'nC')
print ('Rum 4 Dose =', "%.2f" % D4, 'mGy')
print ()
print ()

A = np.array([avg1, avg2, avg3, avg4])

avg_box = np.mean(A)
std_box = np.std(A)

D = avg_box*K

print ('Box gennemsnit =', "%.2f" % avg_box, '±', "%.2f" % std_box, 'nC')
print ()
print ('Dosen er i gennemsnit for hele brættet = ', "%.2f" % D, 'mGy')
print ()
print()

plt.plot(a1*K/1000)
plt.plot(a2*K/1000)
plt.plot(a3*K/1000)
plt.plot(a4*K/1000)
plt.title('Doser målt for %i sekunders eksponering' %t )
plt.legend(['Rum 1', 'Rum 2', 'Rum 3', 'Rum 4'])
plt.ylabel('Dose (Gy)')
plt.xlabel('# Måling')
# plt.show()

# ############################


###############################
