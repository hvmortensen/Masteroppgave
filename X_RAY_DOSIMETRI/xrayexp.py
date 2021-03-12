import numpy as np
import matplotlib.pyplot as plt


#omregningsfaktor for dosimeteret i mGy/nC (milliGray pr nanoCoulomb)
ku = 1
muen_over_rho = 1.075
pu = 1.02
kTP = 1.018
NK = 43.77
K = NK*ku*muen_over_rho*pu*kTP


def CHRG(Dose):
    return Dose/K
print (CHRG(500))

# ## Målinger for t sekunder
# ## Celleflaskebeholderen blev skubbet lidt rundt for at finde det bedste sted:
# ## + 13 cm fra kanten af pladen ved døren og - 13 cm fra højre kant
# ## Tape med sort streg markerer positionen og flaskespidserne peger i y-retning



# ## SSD50
# #05.03.21 efter rep
# t = 17
# a1 = np.array([1.99, 1.98, 1.98, 1.90, 1.93, 1.91, 1.89, 1.91, 1.90, 1.88])
# a2 = np.array([1.90, 1.94, 1.97, 1.90, 1.91, 1.97, 1.92, 1.94, 1.93, 1.97])
# a3 = np.array([1.88, 1.85, 1.86, 1.95, 1.88, 1.88, 1.87, 1.97, 1.85, 1.86])
# a4 = np.array([1.96, 1.94, 1.97, 1.98, 1.87, 1.91, 1.90, 1.92, 1.95, 1.91])

# #
# t = 18
# a1 = np.array([2.12, 2.11, 2.04, 2.10, 2.12, 2.07, 2.10, 2.11, 2.11, 2.12])
# a2 = np.array([2.10, 2.00, 2.13, 2.06, 1.99, 2.12, 2.06, 2.11, 2.07, 2.13])
# a3 = np.array([2.06, 2.04, 2.01, 2.06, 2.08, 2.07, 2.01, 1.98, 1.97, 2.11])
# a4 = np.array([2.02, 2.03, 2.09, 2.06, 2.08, 2.05, 1.99, 2.00, 2.07, 2.03])
#
#

#
# # 05.01.21 før rep
# t = 19
# a1 = np.array([2.20, 2.31, 2.15, 2.25, 2.22, 2.26, 2.21, 2.23, 2.29, 2.23])
# a2 = np.array([2.32, 2.32, 2.26, 2.25, 2.28, 2.30, 2.26, 2.27, 2.18, 2.27])
# a3 = np.array([2.21, 2.20, 2.26, 2.16, 2.15, 2.30, 2.28, 2.21, 2.23, 2.21])
# a4 = np.array([2.18, 2.32, 2.15, 2.18, 2.19, 2.18, 2.28, 2.19, 2.17, 2.22])
# #

#
# t = 20
# a1 = np.array([2.35, 2.46, 2.45, 2.43, 2.39, 2.40, 2.45, 2.39, 2.36, 2.34])
# a2 = np.array([2.47, 2.40, 2.35, 2.39, 2.39, 2.36, 2.44, 2.39, 2.36, 2.43])
# a3 = np.array([2.38, 2.36, 2.32, 2.42, 2.38, 2.36, 2.39, 2.44, 2.34, 2.41])
# a4 = np.array([2.36, 2.34, 2.31, 2.41, 2.31, 2.39, 2.41, 2.41, 2.40, 2.41])
#
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


# # 05.03.21 efter rep
# t = 32
# a1 = np.array([4.01, 4.01, 4.03, 3.98, 4.08, 4.01, 3.98, 3.99, 4.00, 4.03])
# a2 = np.array([4.10, 3.97, 4.02, 4.06, 4.06, 3.97, 3.98, 4.05, 4.02, 4.05])
# a3 = np.array([3.95, 4.00, 4.09, 4.04, 4.00, 4.00, 4.05, 4.02, 4.06, 4.05])
# a4 = np.array([3.97, 4.08, 3.98, 3.98, 4.09, 4.01, 4.03, 4.04, 3.97, 3.96])

# t = 33
# a1 = np.array([4.14, 4.14, 4.24, 4.20, 4.15, 4.25, 4.23, 4.20, 4.16, 4.17])
# a2 = np.array([4.25, 4.22, 4.22, 4.13, 4.18, 4.16, 4.11, 4.18, 4.22, 4.13])
# a3 = np.array([4.14, 4.05, 4.07, 4.15, 4.18, 4.07, 4.16, 4.09, 4.08, 4.19])
# a4 = np.array([4.12, 4.18, 4.12, 4.21, 4.10, 4.20, 4.20, 4.10, 4.11, 4.16])
# #
#

# # 05.03.21 efter rep
# t = 47
# a1 = np.array([6.11, 6.15, 6.13, 6.22, 6.21, 6.11, 6.12, 6.17, 6.15, 6.09])
# a2 = np.array([6.24, 6.09, 6.14, 6.11, 6.12, 6.16, 6.16, 6.08, 6.14, 6.20])
# a3 = np.array([6.16, 6.10, 6.11, 6.12, 6.07, 6.16, 6.07, 6.11, 6.03, 6.14])
# a4 = np.array([6.12, 6.09, 6.07, 6.11, 6.03, 6.12, 6.08, 6.03, 6.12, 6.05])

# t = 48
# a1 = np.array([])
# a2 = np.array([])
# a3 = np.array([])
# a4 = np.array([6.28, 6.24, 6.29, 6.31, 6.23, 6.18, 6.23, 6.22, 6.29, 6.18])
# #




# # # 05.01.21 før rep
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

#
# # 05.01.21 før rep
# t = 53
# a1 = np.array([7.14, 7.09, 7.12, 7.06, 6.99])
# a2 = np.array([7.09, 7.06, 6.99, 6.98, 7.03])
# a3 = np.array([7.06, 6.99, 6.99, 7.06, 7.03])
# a4 = np.array([7.02, 7.03, 7.07, 7.06, 6.93])
#


# # 05.03.21 efter rep
#
# t = 76
# a1 = np.array([10.32, 10.16, 10.28, 10.21, 10.18, 10.19, 10.26, 10.22, 10.20, 10.19])
# a2 = np.array([10.23, 10.15, 10.23, 10.23, 10.18, 10.19, 10.14, 10.14, 10.18, 10.25])
# a3 = np.array([10.16, 10.16, 10.14, 10.23, 10.07, 10.13, 10.06, 10.13, 10.14, 10.14])
# a4 = np.array([10.21, 10.22, 10.11, 10.15, 10.16, 10.24, 10.18, 10.11, 10.22, 10.16])
# #
# #
# t = 77
# a1 = np.array([])
# a2 = np.array([])
# a3 = np.array([10.24, 10.35, 10.26, 10.29, 10.31])
# a4 = np.array([10.27, 10.37, 10.31, 10.26, 10.28, 10.30, 10.34, 10.32, 10.29, 10.37])


# # 05.01.21 før rep
# t = 85
# a1 = np.array([11.50, 11.48, 11.59, 11.53, 11.53])
# a2 = np.array([11.56, 11.54, 11.56, 11.55, 11.45])
# a3 = np.array([11.40, 11.39, 11.47, 11.36, 11.39])
# a4 = np.array([11.50, 11.39, 11.44, 11.49, 11.45])
# #


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

# t = 10
# a1 = np.array([1.46, 1.54, 1.52, 1.53, 1.51])
# a2 = np.array([1.43, 1.46, 1.41, 1.42, 1.37])
# a3 = np.array([1.43, 1.31, 1.31])
# a4 = np.array([])
# # #
# t = 12
# a1 = np.array([1.90])
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
#
#
# t = 22
# a1 = np.array([4.02, 4.06, 3.97, 3.96])
# a2 = np.array([4.11, 4.03, 3.98, 3.94])
# a3 = np.array([])
# a4 = np.array([])
#
# t = 23
# a1 = np.array([4.33, 4.34, 4.16])
# a2 = np.array([4.19, 4.18, 4.14])
# a3 = np.array([])
# a4 = np.array([])
# t = 24
# a1 = np.array([4.55, 4.56])
# a2 = np.array([])
# a3 = np.array([])
# a4 = np.array([])
# t = 26
# a1 = np.array([5.01])
# a2 = np.array([])
# a3 = np.array([])
# a4 = np.array([])
# t = 31
# a1 = np.array([6.00, 6.04, 5.88])
# a2 = np.array([5.89])
# a3 = np.array([])
# a4 = np.array([])
# t = 32
# a1 = np.array([6.32, 6.15, 6.23])
# a2 = np.array([6.25, 6.13, 6.05])
# a3 = np.array([])
# a4 = np.array([])
# t = 51
# a1 = np.array([10.30])
# a2 = np.array([10.17])
# a3 = np.array([])
# a4 = np.array([])

# t = 52
# a1 = np.array([10.48])
# a2 = np.array([])
# a3 = np.array([])
# a4 = np.array([])

# t = 97
# a1 = np.array([20.25])
# a2 = np.array([20.09])
# a3 = np.array([])
# a4 = np.array([])

# t = 98
# a1 = np.array([20.68])
# a2 = np.array([20.37])
# a3 = np.array([])
# a4 = np.array([])
# t = 102
# a1 = np.array([21.34])
# a2 = np.array([])
# a3 = np.array([])
# a4 = np.array([])





# #### SSD37_5 ####
# t = 12
# a1 = np.array([2.05, 2.04, 2.04, 1.95, 1.95])
# a2 = np.array([2.01, 2.07, 1.92, 2.00, 2.06])
# a3 = np.array([1.96, 2.05, 1.99, 2.05, 2.03])
# a4 = np.array([2.12, 2.00, 2.06, 1.90, 1.94])
# #
# t = 13
# a1 = np.array([])
# a2 = np.array([])
# a3 = np.array([])
# a4 = np.array([2.17, 2.29, 2.24, 2.23, 2.10])
# #

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
# A = np.array([a1, a2, a3, a4])


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
