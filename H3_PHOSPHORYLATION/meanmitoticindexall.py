import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'cm'
mpl.rcParams['font.sans-serif'] = 'Times New Roman'
mpl.rcParams['font.family'] = 'Times New Roman'

D = np.array([0, 0.1, 0.2, 0.3, 0.5, 1.0])

data = np.loadtxt("H3_alldata.txt")
M = data

# print(M[1][0])
# print(M[0,1:])
h0 = []
h1 = []
h2 = []
h3 = []
h5 = []
hx = []
p0 = []
p1 = []
p2 = []
p3 = []
p5 = []
px = []
t0 = []
t1 = []
t2 = []
t3 = []
t5 = []
# tx = []
for i in range(int(M.shape[0])):
    if M[i][0]==10:
        h0.append(np.mean(M[i,1:]))
    elif M[i][0]==11:
        h1.append(np.mean(M[i,1:]))
    elif M[i][0]==12:
        h2.append(np.mean(M[i,1:]))
    elif M[i][0]==13:
        h3.append(np.mean(M[i,1:]))
    elif M[i][0]==15:
        h5.append(np.mean(M[i,1:]))
    elif M[i][0]==110:
        hx.append(np.mean(M[i,1:]))

    elif M[i][0]==20:
        p0.append(np.mean(M[i,1:]))
    elif M[i][0]==21:
        p1.append(np.mean(M[i,1:]))
    elif M[i][0]==22:
        p2.append(np.mean(M[i,1:]))
    elif M[i][0]==23:
        p3.append(np.mean(M[i,1:]))
    elif M[i][0]==25:
        p5.append(np.mean(M[i,1:]))
    elif M[i][0]==210:
        px.append(np.mean(M[i,1:]))

    elif M[i][0]==30:
        t0.append(np.mean(M[i,1:]))
    elif M[i][0]==31:
        t1.append(np.mean(M[i,1:]))
    elif M[i][0]==32:
        t2.append(np.mean(M[i,1:]))
    elif M[i][0]==33:
        t3.append(np.mean(M[i,1:]))
    elif M[i][0]==35:
        t5.append(np.mean(M[i,1:]))
    # elif M[i][0]==310:
    #     tx.append(np.mean(M[i,1:]))
#
# print(len(hx))
# print(h0)
# print(hx)

print(hx[0]/h0[0])
print(hx[1]/h0[1])
print(hx[2]/h0[2])
print()
print(h1[0]/h0[0])
print()



hm0 = np.zeros_like(h0)
hm1 = np.zeros_like(h1)
hm2 = np.zeros_like(h2)
hm3 = np.zeros_like(h3)
hm5 = np.zeros_like(h5)
hmx = np.zeros_like(hx)
pm0 = np.zeros_like(p0)
pm1 = np.zeros_like(p1)
pm2 = np.zeros_like(p2)
pm3 = np.zeros_like(p3)
pm5 = np.zeros_like(p5)
pmx = np.zeros_like(px)
tm0 = np.zeros_like(t0)
tm1 = np.zeros_like(t1)
tm2 = np.zeros_like(t2)
tm3 = np.zeros_like(t3)
tm5 = np.zeros_like(t5)
# tmx = np.zeros_like(tx)
for i in range(len(h0)):
    hm0[i] = (h0[i])/(h0[i])
for i in range(len(h1)):
    hm1[i] = (h1[i])/(h0[i])
for i in range(len(h2)):
    hm2[i] = (h2[i])/(h0[i])
for i in range(len(h3)):
    hm3[i] = (h3[i])/(h0[i])
for i in range(len(h5)):
    hm5[i] = (h5[i])/(h0[i])
for i in range(len(hx)):
    hmx[i] = (hx[i])/(h0[i])

for i in range(len(p0)):
    pm0[i] = (p0[i])/(p0[i])
for i in range(len(p1)):
    pm1[i] = (p1[i])/(p0[i])
for i in range(len(p2)):
    pm2[i] = (p2[i])/(p0[i])
for i in range(len(p3)):
    pm3[i] = (p3[i])/(p0[i])
for i in range(len(p5)):
    pm5[i] = (p5[i])/(p0[i])
for i in range(len(px)):
    pmx[i] = (px[i])/(p0[i])

for i in range(len(t0)):
    tm0[i] = (t0[i])/(t0[i])
for i in range(len(t1)):
    tm1[i] = (t1[i])/(t0[i])
for i in range(len(t2)):
    tm2[i] = (t2[i])/(t0[i])
for i in range(len(t3)):
    tm3[i] = (t3[i])/(t0[i])
for i in range(len(t5)):
    tm5[i] = (t5[i])/(t0[i])
# for i in range(len(tx)):
#     tmx[i] = (tx[i])/(t0[i])

hm = np.array([np.mean(hm0), np.mean(hm1), np.mean(hm2), np.mean(hm3), np.mean(hm5), np.mean(hmx)])
pm = np.array([np.mean(pm0), np.mean(pm1), np.mean(pm2), np.mean(pm3), np.mean(pm5), np.mean(pmx)])
tm = np.array([np.mean(tm0), np.mean(tm1), np.mean(tm2), np.mean(tm3), np.mean(tm5)])

print(hm)

plt.plot(D,hm)
plt.plot(D,pm)
plt.plot(D[:-1],tm)

plt.show()
