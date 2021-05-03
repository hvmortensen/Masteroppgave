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
NK = 43.77
K = NK*ku*muen_over_rho*pu*kTP


A18 = np.array([2.12, 2.11, 2.04, 2.10, 2.12, 2.07, 2.10, 2.11, 2.11, 2.12, 2.10, 2.00, 2.13, 2.06, 1.99, 2.12, 2.06, 2.11, 2.07, 2.13, 2.06, 2.04, 2.01, 2.06, 2.08, 2.07, 2.01, 1.98, 1.97, 2.11, 2.02, 2.03, 2.09, 2.06, 2.08, 2.05, 1.99, 2.00, 2.07, 2.03])
A33 = np.array([4.14, 4.14, 4.24, 4.20, 4.15, 4.25, 4.23, 4.20, 4.16, 4.17, 4.25, 4.22, 4.22, 4.13, 4.18, 4.16, 4.11, 4.18, 4.22, 4.13, 4.14, 4.05, 4.07, 4.15, 4.18, 4.07, 4.16, 4.09, 4.08, 4.19, 4.12, 4.18, 4.12, 4.21, 4.10, 4.20, 4.20, 4.10, 4.11, 4.16])
A47 = np.array([6.11, 6.15, 6.13, 6.22, 6.21, 6.11, 6.12, 6.17, 6.15, 6.09, 6.24, 6.09, 6.14, 6.11, 6.12, 6.16, 6.16, 6.08, 6.14, 6.20, 6.16, 6.10, 6.11, 6.12, 6.07, 6.16, 6.07, 6.11, 6.03, 6.14, 6.12, 6.09, 6.07, 6.11, 6.03, 6.12, 6.08, 6.03, 6.12, 6.05])
A76 = np.array([10.32, 10.16, 10.28, 10.21, 10.18, 10.19, 10.26, 10.22, 10.20, 10.19, 10.23, 10.15, 10.23, 10.23, 10.18, 10.19, 10.14, 10.14, 10.18, 10.25, 10.16, 10.16, 10.14, 10.23, 10.07, 10.13, 10.06, 10.13, 10.14, 10.14, 10.21, 10.22, 10.11, 10.15, 10.16, 10.24, 10.18, 10.11, 10.22, 10.16])

B13 = np.array([2.20, 2.04, 2.00, 1.98, 2.16, 2.17, 1.99, 2.04, 2.11, 2.05, 2.14, 1.99, 1.94, 2.05, 2.05, 2.22, 2.09, 1.96, 2.05, 1.95])
B22 = np.array([4.02, 4.06, 3.97, 3.96, 4.11, 4.03, 3.98, 3.94])
B32 = np.array([6.32, 6.15, 6.23, 6.25, 6.13, 6.05])
B51 = np.array([10.30, 10.17])
B98 = np.array([20.68, 20.37])

a18 = np.mean(A18)*K;a33 = np.mean(A33)*K;a47 = np.mean(A47)*K;a76 = np.mean(A76)*K;b13 = np.mean(B13)*K;b22 = np.mean(B22)*K;b32 = np.mean(B32)*K;b51 = np.mean(B51)*K;b98 = np.mean(B98)*K
as18 = np.std(A18)*K;as33 = np.std(A33)*K;as47 = np.std(A47)*K;as76 = np.std(A76)*K;bs13 = np.std(B13)*K;bs22 = np.std(B22)*K;bs32 = np.std(B32)*K;bs51 = np.std(B51)*K;bs98 = np.std(B98)*K

ac = np.array([18, 33, 47, 76])
bc = np.array([13, 22, 32, 51, 98])
a = np.array([a18,a33,a47,a76])
b = np.array([b13,b22,b32,b51,b98])
a_s = np.array([as18,as33,as47,as76])
b_s = np.array([bs13,bs22,bs32,bs51,bs98])

al = np.linspace(13,98,1001)

FS = 17+3; fs = FS - 2  # fontsize til lengend()

fl = np.polyfit(bc,b,1)
m,l = fl
def f(x):
    return m*x + l
#
hl = np.polyfit(ac,a,1)
c,d = hl
def g(x):
    return c*x + d


# print(d)
# print(l)
# print(c)
# print(m)
#
for i in range(100):
    print(i, g(i)/f(i))




fig, ax = plt.subplots(1,2, figsize=(13,4.8))

ax[0].plot(bc,b,"bo")
ax[0].plot(al,f(al),label="$f(t)=%.2ft%.2f$"%(m,l))
ax[0].plot(ac,a, "ro")#, label="Doser, SSD50")
ax[0].plot(al,g(al),label="$g(t)=%.2ft%.2f$"%(c,d))
ax[0].set_xlabel("Eksponeringstid (s)",fontsize=FS)
ax[0].set_ylabel("Dosis (mGy)",fontsize=FS)
ax[0].set_xticks([10,20,30,40,50,60,70,80,90,100])
ax[0].set_yticks([100,200,300,400,500,600,700,800,900,1000])
ax[0].tick_params(axis='both', which='major', labelsize=FS)
ax[0].legend(fontsize=FS)
ax[0].text(49,220 ,"Doser, SSD50",fontsize=FS)
ax[0].text(20,500 ,"Doser, SSD40",fontsize=FS)

x = 1; y = 0.3 # forskydning til labels

ax[1].plot(bc,b/bc)#, label="SSD = 40 cm")
ax[1].plot(ac,a/ac)#, label="SSD = 50 cm")
ax[1].plot(bc,b/bc, "bo", label="Doser, SSD40")
ax[1].plot(ac,a/ac, "ro", label="Doser, SSD50")
ax[1].text(13+x,b13/13-y ,'0.1 Gy',fontsize=fs)
ax[1].text(22+x,b22/22-y ,'0.2 Gy',fontsize=fs)
ax[1].text(32+x,b32/32-y ,'0.3 Gy',fontsize=fs)
ax[1].text(51+x,b51/51-y ,'0.5 Gy',fontsize=fs)
ax[1].text(98+4,b98/98-y-0.1 ,'1.0 Gy',horizontalalignment='right',fontsize=fs)
ax[1].text(18+x+3,a18/18-y+0.2 ,'0.1 Gy',fontsize=fs)
ax[1].text(33+x,a33/33-y ,'0.2 Gy',fontsize=fs)
ax[1].text(47+x,a47/47-y ,'0.3 Gy',fontsize=fs)
ax[1].text(76+x,a76/76-y ,'0.5 Gy',fontsize=fs)
ax[1].set_xlabel("Eksponeringstid (s)",fontsize=FS)
ax[1].set_ylabel("Doserate (mGy/s)",fontsize=FS)
ax[1].set_xticks([10,20,30,40,50,60,70,80,90,100])
ax[1].set_yticks([5.5,6,6.5,7,7.5,8,8.5,9,9.5,10])
ax[1].tick_params(axis='both', which='major', labelsize=FS)
ax[1].text(40,6.7 ,"Doser, SSD50",fontsize=FS)
ax[1].text(20,10 ,"Doser, SSD40",fontsize=FS)

plt.tight_layout()
plt.savefig("Doserate.pdf")
plt.show()

r1 = 40
r2 = 50
konv = (f(20000000)*r1**2)/(g(20000000)*r2**2)
FS = FS - 3
plt.plot(al, (f(al)*r1**2)/(g(al)*r2**2), label="$\\frac{I_1(t)\\cdot r_1^2}{I_2(t)\\cdot r_2^2}=\\frac{f(t)\\cdot (40cm)^2}{g(t)\\cdot (50cm)^2}\\approx1$")
plt.axhline(konv,linestyle='--',color='red')
plt.tick_params(axis='both', which='major', labelsize=FS)
plt.text(20,0.99,"asymptote ≈ %.3f"%konv,fontsize=FS )
plt.xlabel("Eksponeringstid (s)",fontsize=FS)
plt.xticks([10,20,30,40,50,60,70,80,90,100])
plt.legend(fontsize=(FS+2))
plt.tight_layout()
plt.show()


al = np.linspace(0,98)
FS = FS - 3
plt.plot(al,f(al),label="$f(t)=%.2ft%.2f = 0 \\Rightarrow t = %.2fs$"%(m,l,-l/m))
plt.plot(al,g(al),label="$g(t)=%.2ft%.2f = 0 \\Rightarrow t = %.2fs$"%(c,d,-d/c))
plt.xlabel("Eksponeringstid (s)",fontsize=FS)
plt.ylabel("Dosis (mGy)",fontsize=FS)
plt.ylim(-40,60)
plt.xlim(0,8)
plt.tick_params(axis='both', which='major', labelsize=FS)
plt.legend(fontsize=FS)
plt.tight_layout()
plt.show()






# print(a18)
# print(a33)
# print(a47)
# print(a76)
# print(b13)
# print(b22)
# print(b32)
# print(b51)
# print(b98)

### Alle dosegennemsnit fra xrayexp.py ###

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
# bbs = np.array([b10,b12,b13,b14,b15,b22,b23,b24,b31,b32,b51,b52,b97,b98,b102])
# bsc = np.array([10,12,13,14,15,22,23,24,31,32,51,52,97,98,102])
