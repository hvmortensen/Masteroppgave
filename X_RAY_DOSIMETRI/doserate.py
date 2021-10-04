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


### A = SSD50
A18 = np.array([2.12, 2.11, 2.04, 2.10, 2.12, 2.07, 2.10, 2.11, 2.11, 2.12, 2.10, 2.00, 2.13, 2.06, 1.99, 2.12, 2.06, 2.11, 2.07, 2.13, 2.06, 2.04, 2.01, 2.06, 2.08, 2.07, 2.01, 1.98, 1.97, 2.11, 2.02, 2.03, 2.09, 2.06, 2.08, 2.05, 1.99, 2.00, 2.07, 2.03])
A33 = np.array([4.14, 4.14, 4.24, 4.20, 4.15, 4.25, 4.23, 4.20, 4.16, 4.17, 4.25, 4.22, 4.22, 4.13, 4.18, 4.16, 4.11, 4.18, 4.22, 4.13, 4.14, 4.05, 4.07, 4.15, 4.18, 4.07, 4.16, 4.09, 4.08, 4.19, 4.12, 4.18, 4.12, 4.21, 4.10, 4.20, 4.20, 4.10, 4.11, 4.16])
A47 = np.array([6.11, 6.15, 6.13, 6.22, 6.21, 6.11, 6.12, 6.17, 6.15, 6.09, 6.24, 6.09, 6.14, 6.11, 6.12, 6.16, 6.16, 6.08, 6.14, 6.20, 6.16, 6.10, 6.11, 6.12, 6.07, 6.16, 6.07, 6.11, 6.03, 6.14, 6.12, 6.09, 6.07, 6.11, 6.03, 6.12, 6.08, 6.03, 6.12, 6.05])
A76 = np.array([10.32, 10.16, 10.28, 10.21, 10.18, 10.19, 10.26, 10.22, 10.20, 10.19, 10.23, 10.15, 10.23, 10.23, 10.18, 10.19, 10.14, 10.14, 10.18, 10.25, 10.16, 10.16, 10.14, 10.23, 10.07, 10.13, 10.06, 10.13, 10.14, 10.14, 10.21, 10.22, 10.11, 10.15, 10.16, 10.24, 10.18, 10.11, 10.22, 10.16])


### B = SSD40
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

al1 = np.linspace(0,98,10001)


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
print(al)
# for i in range(100):
#     print(i, g(i)/f(i))



FS = 17+3;
FS = FS
fs = FS  # fontsize til legend()
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
ax[0].legend(fontsize=FS-2)
ax[0].text(49,220 ,"Doser, SSD50",fontsize=FS)
ax[0].text(20,500 ,"Doser, SSD40",fontsize=FS)


x = 1; y = 0.3 # forskydning til labels

# al = np.linspace(0,98)
ax[1].plot(al1,f(al1),label="$f(t)= 0 \\Rightarrow t = %.2f$ s"%(-l/m))
ax[1].plot(al1,g(al1),label="$g(t)= 0 \\Rightarrow t = %.2f$ s"%(-d/c))
ax[1].plot( (d - l)/(m - c),f((d - l)/(m - c)),"k.", label="$f(t) = g(t)\\Rightarrow t = %.2f$ s"%( (d - l)/(m - c)) )
ax[1].axhline(0, linestyle='--', color='k')
ax[1].set_xlabel("Eksponeringstid (s)",fontsize=FS)
# ax[1].set_ylabel("Dosis (mGy)",fontsize=FS)
ax[1].set_ylim(-40,60)
ax[1].set_xlim(0,8)
ax[1].set_xticks([0,1,2,3,4,5,6,7,8])
ax[1].tick_params(axis='both', which='major', labelsize=FS)
ax[1].legend(fontsize=FS-2)
plt.tight_layout()
plt.savefig("Dosisskaeringspunkt.pdf")
plt.show()

SSD40data = np.loadtxt("SSD40_1s-13s_data.txt")
SSD50data = np.loadtxt("SSD50_1s-18s_data.txt")

M = SSD40data
N = SSD50data

m40 = np.zeros(M.shape[0])
s40 = np.zeros(M.shape[0])
for i in range(M.shape[0]):
    m40[i] = np.mean(M[i])*K
    s40[i] = np.std(M[i])*K
m50 = np.zeros(N.shape[0])
s50 = np.zeros(N.shape[0])
for i in range(N.shape[0]):
    m50[i] = np.mean(N[i])*K
    s50[i] = np.std(N[i])*K

t40 = np.array([1,2,3,4,5,6,7,8,9,10, 11, 12, 13 ])
t50 = np.array([1,2,3,4,5,6,7,8,9,10, 11, 12, 13,14,15,16,17,18 ])
tl = np.linspace(0,18,1001)

ql = np.linspace(4,18,1001)


start = 4
slut = 9

hl = np.polyfit(t40[start:],m40[start:],1)
o,p = hl
def h(x):
    return o*x + p
#
il = np.polyfit(t50[start:],m50[start:],1)
y,u = il
def i(x):
    return y*x + u

# FS = FS - 2  # fontsize til lengend()
figa, axa = plt.subplots(1,2, figsize=(13,4.8))
axa[0].plot(t40, m40, "bo")
axa[0].plot(ql, h(ql),label="$f(t)=%.2ft%.2f$"%(o,p))
axa[0].plot(t50, m50, "ro")
axa[0].plot(ql, i(ql),label="$g(t)=%.2ft%.2f$"%(y,u))
axa[0].axhline(0, linestyle='--', color='k')
axa[0].set_xlabel("Eksponeringstid (s)",fontsize=FS)
axa[0].set_ylabel("Dosis (mGy)",fontsize=FS)
axa[0].set_yticks([0, 25, 50, 75, 100, 125, 150])
axa[0].set_xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])
axa[0].tick_params(axis='both', which='major', labelsize=FS)
axa[0].legend(fontsize=FS-2)
axa[0].text(2.8,65 ,"Doser, SSD40",fontsize=FS)
axa[0].text(7,10 ,"Doser, SSD50",fontsize=FS)
# plt.tight_layout()
# plt.show()
# m = o
# l = p
# c = y
# d = u
axa[1].plot(tl,h(tl),label="$f(t)= 0 \\Rightarrow t = %.2f$ s"%(-p/o))
axa[1].plot(tl,i(tl),label="$g(t)= 0 \\Rightarrow t = %.2f$ s"%(-u/y))
axa[1].plot( (u - p)/(o - y),h((u - p)/(o - y)),"k.", label="$f(t) = g(t)\\Rightarrow t = %.2f$ s"%((u - p)/(o - y)) )
axa[1].axhline(0, linestyle='--', color='k')
axa[1].set_xlabel("Eksponeringstid (s)",fontsize=FS)
# axa[1].set_ylabel("Dosis (mGy)",fontsize=FS)
axa[1].set_ylim(-40,60)
axa[1].set_xlim(0,8)
axa[1].set_xticks([0,1,2,3,4,5,6,7,8])
axa[1].tick_params(axis='both', which='major', labelsize=FS)
axa[1].legend(fontsize=FS-2)
plt.tight_layout()
plt.savefig("mikrodosisskaeringspunkt.pdf")
plt.show()



# plt.plot(bc,b/(bc-3.66))#, label="SSD = 40 cm")
# plt.plot(ac,a/(ac-3.66))#, label="SSD = 50 cm")
# plt.plot(bc,b/(bc-3.66), "bo", label="Doser, SSD40")
# plt.plot(ac,a/(ac-3.66), "ro", label="Doser, SSD50")
# plt.text(13+x,b13/13-y ,'0.1 Gy',fontsize=fs)
# plt.text(22+x,b22/22-y ,'0.2 Gy',fontsize=fs)
# plt.text(32+x,b32/32-y ,'0.3 Gy',fontsize=fs)
# plt.text(51+x,b51/51-y ,'0.5 Gy',fontsize=fs)
# plt.text(98+4,b98/98-y-0.1 ,'1.0 Gy',horizontalalignment='right',fontsize=fs)
# plt.text(18+x+3,a18/18-y+0.2 ,'0.1 Gy',fontsize=fs)
# plt.text(33+x,a33/33-y ,'0.2 Gy',fontsize=fs)
# plt.text(47+x,a47/47-y ,'0.3 Gy',fontsize=fs)
# plt.text(76+x,a76/76-y ,'0.5 Gy',fontsize=fs)
# plt.text(40,6.7 ,"Doser, SSD50",fontsize=FS)
# plt.text(20,10 ,"Doser, SSD40",fontsize=FS)
# plt.xlabel("Eksponeringstid (s)",fontsize=FS)
# plt.ylabel("Doserate (Gy/time)",fontsize=FS)
# plt.xticks([10,20,30,40,50,60,70,80,90,100])
# plt.yticks([5.5,6,6.5,7,7.5,8,8.5,9,9.5,10])
# plt.tick_params(axis='both', which='major', labelsize=FS)
# plt.tight_layout()
# plt.savefig("Doserate1.pdf")
# plt.show()

plex = 1.5
kav = 0.31
fejl = 2.5

r1 = 40 - plex - kav
r2 = 50 - plex - kav 
konv = (f(20000000)*r1**2)/(g(20000000)*r2**2)

print(al)
FS = FS -3
plt.plot(al, (f(al)*r1**2)/(g(al)*r2**2), label="$\\frac{D_{40}(t)\\cdot SSD40^2}{D_{50}(t)\\cdot SSD50^2}=\\frac{f(t)\\cdot (40 - 1.5 - 0.31)^2}{g(t)\\cdot (50 - 1.5 - 0.31)^2}\\approx1$")
plt.axhline(konv,linestyle='--',color='red')#, label="asymptote ≈ %.3f"%konv)
plt.tick_params(axis='both', which='major', labelsize=FS)
plt.text(12,0.972,"asymptote ≈ %.3f"%konv,fontsize=FS )
plt.xlabel("Eksponeringstid (s)",fontsize=FS)
plt.xticks([10,20,30,40,50,60,70,80,90,100])
plt.legend(fontsize=(FS+2))
plt.tight_layout()
plt.savefig("Afst_kvadr_lov.pdf")
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
