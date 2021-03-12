import numpy as np
import matplotlib.pyplot as plt

ku = 1
muen_over_rho = 1.075
pu = 1.02
kTP = 1.018
NK = 43.77
K = NK*ku*muen_over_rho*pu*kTP

# # # SSD50
# a17 = 93.78
# a18 = 100.87
# a19 = 109.11
# a20 = 116.71
# a32 = 196.37
# a33 = 203.20
# a35 = 219.60
# a36 = 225.71
# a47 = 299.00
# a48 = 305.12
# a51 = 327.86
# a52 = 336.87
# a53 = 343.93
# a76 = 497.43
# a77 = 503.26
# a85 = 560.81
#
# # # SSD40
# b10 = 69.70
# b12 = 91.92
# b13 = 100.60
# b14 = 108.15
# b15 = 118.90
# b22 = 195.85
# b23 = 206.35
# b24 = 222.55
# b31 = 290.83
# b32 = 302.35
# b51 = 500.06
# b52 = 512.03
# b97 = 985.46
# b98 = 1000.80
# b102 = 1042.62

# aas = np.array([a17,a18,a19,a20,a32,a35,a36,a47,a48,a51,a52,a53,a76,a77,a85])
# asc = np.array([17,18,19,20,32,35,36,47,48,51,52,53,76,77,85])
#
# bbs = np.array([b10,b12,b13,b14,b15,b22,b23,b24,b31,b32,b51,b52,b97,b98,b102])
# bsc = np.array([10,12,13,14,15,22,23,24,31,32,51,52,97,98,102])

A18 = np.array([2.12, 2.11, 2.04, 2.10, 2.12, 2.07, 2.10, 2.11, 2.11, 2.12, 2.10, 2.00, 2.13, 2.06, 1.99, 2.12, 2.06, 2.11, 2.07, 2.13, 2.06, 2.04, 2.01, 2.06, 2.08, 2.07, 2.01, 1.98, 1.97, 2.11, 2.02, 2.03, 2.09, 2.06, 2.08, 2.05, 1.99, 2.00, 2.07, 2.03])
A33 = np.array([4.14, 4.14, 4.24, 4.20, 4.15, 4.25, 4.23, 4.20, 4.16, 4.17, 4.25, 4.22, 4.22, 4.13, 4.18, 4.16, 4.11, 4.18, 4.22, 4.13, 4.14, 4.05, 4.07, 4.15, 4.18, 4.07, 4.16, 4.09, 4.08, 4.19, 4.12, 4.18, 4.12, 4.21, 4.10, 4.20, 4.20, 4.10, 4.11, 4.16])
A47 = np.array([6.11, 6.15, 6.13, 6.22, 6.21, 6.11, 6.12, 6.17, 6.15, 6.09, 6.24, 6.09, 6.14, 6.11, 6.12, 6.16, 6.16, 6.08, 6.14, 6.20, 6.16, 6.10, 6.11, 6.12, 6.07, 6.16, 6.07, 6.11, 6.03, 6.14, 6.12, 6.09, 6.07, 6.11, 6.03, 6.12, 6.08, 6.03, 6.12, 6.05])
A76 = np.array([10.32, 10.16, 10.28, 10.21, 10.18, 10.19, 10.26, 10.22, 10.20, 10.19, 10.23, 10.15, 10.23, 10.23, 10.18, 10.19, 10.14, 10.14, 10.18, 10.25, 10.16, 10.16, 10.14, 10.23, 10.07, 10.13, 10.06, 10.13, 10.14, 10.14, 10.21, 10.22, 10.11, 10.15, 10.16, 10.24, 10.18, 10.11, 10.22, 10.16])

B13 = np.array([2.20, 2.04, 2.00, 1.98, 2.16, 2.17, 1.99, 2.04, 2.11, 2.05, 2.14, 1.99, 1.94, 2.05, 2.05, 2.22, 2.09, 1.96, 2.05, 1.95])
B22 = np.array([4.02, 4.06, 3.97, 3.96, 4.11, 4.03, 3.98, 3.94])
B32 = np.array([6.32, 6.15, 6.23, 6.25, 6.13, 6.05])
B51 = np.array([10.30, 10.17])
B98 = np.array([20.68, 20.37])

a18 = np.mean(A18)*K
a33 = np.mean(A33)*K
a47 = np.mean(A47)*K
a76 = np.mean(A76)*K
b13 = np.mean(B13)*K
b22 = np.mean(B22)*K
b32 = np.mean(B32)*K
b51 = np.mean(B51)*K
b98 = np.mean(B98)*K

as18 = np.std(A18)*K
as33 = np.std(A33)*K
as47 = np.std(A47)*K
as76 = np.std(A76)*K
bs13 = np.std(B13)*K
bs22 = np.std(B22)*K
bs32 = np.std(B32)*K
bs51 = np.std(B51)*K
bs98 = np.std(B98)*K

print(as18)
print(as33)
print(as47)
print(as76)
print(bs13)
print(bs22)
print(bs32)
print(bs51)
print(bs98)


ac = np.array([18,33,47,76])
bc = np.array([13,22,32,51,98])
a = np.array([a18,a33,a47,a76])
b = np.array([b13,b22,b32,b51,b98])

x = 1
y = 0.2


FS = 14 # fontsixe til superoverskrift
fs = FS - 2  # fontsize til lengend()
FFS = FS + 3

plt.plot(bc,b/bc, label="SSD = 40 cm")
plt.plot(ac,a/ac, label="SSD = 50 cm")
#
# plt.plot(bsc,bbs/bsc, label="SSD = 40 cm")
# plt.plot(asc,aas/asc, label="SSD = 50 cm")
plt.plot(13,b13/13,"bo")
plt.text(13+x,b13/13-y ,'0.1 Gy',fontsize=fs)# / \n %.2f mGy/s'%(b13/13))

plt.plot(22,b22/22,"bo")
plt.text(22+x,b22/22-y ,'0.2 Gy',fontsize=fs)

plt.plot(32,b32/32,"bo")
plt.text(32+x,b32/32-y ,'0.3 Gy',fontsize=fs)

plt.plot(51,b51/51,"bo")
plt.text(51+x,b51/51-y ,'0.5 Gy',fontsize=fs)

plt.plot(98,b98/98,"bo")
plt.text(98+4,b98/98-0.3 ,'1.0 Gy',horizontalalignment='right',fontsize=fs)

plt.plot(18,a18/18,"ro")
plt.text(18+x,a18/18-y ,'0.1 Gy',fontsize=fs)

plt.plot(33,a33/33,"ro")
plt.text(33+x,a33/33-y ,'0.2 Gy',fontsize=fs)

plt.plot(47,a47/47,"ro")
plt.text(47+x,a47/47-y ,'0.3 Gy',fontsize=fs)

plt.plot(76,a76/76,"ro")
plt.text(76+x,a76/76-y ,'0.5 Gy',fontsize=fs)




plt.xlabel("Tid (s)",fontsize=FS)
plt.ylabel("Doserate (mGy/s)",fontsize=FS)
plt.xticks([10,20,30,40,50,60,70,80,90,100])
plt.yticks([5.5,6,6.5,7,7.5,8,8.5,9,9.5,10])
plt.xticks(fontsize = FS)
plt.yticks(fontsize = FS)
plt.legend(fontsize=FS, loc='center right')
plt.tight_layout()
plt.savefig("Doserate.pdf")
plt.show()
