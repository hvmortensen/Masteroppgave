import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'cm'
mpl.rcParams['font.sans-serif'] = 'Times New Roman'
mpl.rcParams['font.family'] = 'Times New Roman'

D = np.array([0, 0.1, 0.2, 0.3, 0.5])




h180a = np.array([3.3, 4.0, 3.3, 3.0, 2.9])
h181a = np.array([1.7, 2.6, 1.5, 1.6, 1.1])
h182a = np.array([1.9, 2.7, 2.2, 2.1, 1.7])
h183a = np.array([1.7, 2.4, 1.8, 2.0, 1.5])
h185a = np.array([1.4, 2.1, 1.2, 1.2, 1.1])
p180a = np.array([2.2, 2.3, 2.3, 2.4, 2.2])
p181a = np.array([1.8, 2.8, 1.9, 2.1, 1.6])
p182a = np.array([1.8, 2.0, 1.8, 1.9, 1.3])
p183a = np.array([2.2, 2.5, 1.8, 1.6, 1.8])
p185a = np.array([0.9, 1.2, 1.1, 1.0, 0.9])
t180a = np.array([2.2, 2.5, 2.2, 2.4, 2.1])
t181a = np.array([2.3, 2.9, 2.2, 2.7, 1.5])
t182a = np.array([2.6, 2.7, 2.9, 2.1, 1.8])
t183a = np.array([1.4, 1.7, 1.6, 1.5, 1.4])
t185a = np.array([1.9, 2.3, 1.6, 1.6, 1.8])

h180am = np.mean(h180a)
h181am = np.mean(h181a)
h182am = np.mean(h182a)
h183am = np.mean(h183a)
h185am = np.mean(h185a)
p180am = np.mean(p180a)
p181am = np.mean(p181a)
p182am = np.mean(p182a)
p183am = np.mean(p183a)
p185am = np.mean(p185a)
t180am = np.mean(t180a)
t181am = np.mean(t181a)
t182am = np.mean(t182a)
t183am = np.mean(t183a)
t185am = np.mean(t185a)

h180as = np.std(h180a)
h181as = np.std(h181a)
h182as = np.std(h182a)
h183as = np.std(h183a)
h185as = np.std(h185a)
p180as = np.std(p180a)
p181as = np.std(p181a)
p182as = np.std(p182a)
p183as = np.std(p183a)
p185as = np.std(p185a)
t180as = np.std(t180a)
t181as = np.std(t181a)
t182as = np.std(t182a)
t183as = np.std(t183a)
t185as = np.std(t185a)


### RUN 2
h180b = np.array([2.3, 2.7, 2.8, 2.6, 2.6])
h181b = np.array([1.3, 1.4, 1.4, 1.6, 1.4])
h182b = np.array([1.8, 1.8, 1.9, 1.7, 1.9])
h183b = np.array([1.6, 1.5, 1.5, 1.1, 1.4])
h185b = np.array([1.2, 1.0, 1.3, 1.2, 1.1])
p180b = np.array([2.2, 2.2, 2.6, 2.1, 2.2])
p181b = np.array([1.6, 1.4, 1.9, 1.6, 1.9])
p182b = np.array([1.5, 1.3, 1.7, 1.3, 1.5])
p183b = np.array([1.9, 1.6, 1.8, 1.7, 2.1])
p185b = np.array([0.9, 0.8, 0.9, 0.8, 0.8])
t180b = np.array([1.9, 1.9, 2.1, 2.1, 1.9])
t181b = np.array([1.9, 1.7, 2.6, 1.6, 1.9])
t182b = np.array([2.0, 1.7, 2.2, 1.5, 1.5])
t183b = np.array([1.4, 1.2, 1.3, 1.2, 1.4])
t185b = np.array([1.6, 1.4, 1.9, 1.2, 1.4])

h180bm = np.mean(h180b)
h181bm = np.mean(h181b)
h182bm = np.mean(h182b)
h183bm = np.mean(h183b)
h185bm = np.mean(h185b)
p180bm = np.mean(p180b)
p181bm = np.mean(p181b)
p182bm = np.mean(p182b)
p183bm = np.mean(p183b)
p185bm = np.mean(p185b)
t180bm = np.mean(t180b)
t181bm = np.mean(t181b)
t182bm = np.mean(t182b)
t183bm = np.mean(t183b)
t185bm = np.mean(t185b)

h180bs = np.std(h180b)
h181bs = np.std(h181b)
h182bs = np.std(h182b)
h183bs = np.std(h183b)
h185bs = np.std(h185b)
p180bs = np.std(p180b)
p181bs = np.std(p181b)
p182bs = np.std(p182b)
p183bs = np.std(p183b)
p185bs = np.std(p185b)
t180bs = np.std(t180b)
t181bs = np.std(t181b)
t182bs = np.std(t182b)
t183bs = np.std(t183b)
t185bs = np.std(t185b)


### RUN 3 (Jenny)
h180c = np.array([4.02, 3.94, 3.28, 3.60, 3.68])
h181c = np.array([3.48, 3.21, 2.83, 2.95, 3.06])
h182c = np.array([3.43, 2.43, 2.42, 2.85, 2.88])
h183c = np.array([3.05, 2.64, 2.40, 2.65, 2.73])
h185c = np.array([3.19, 2.67, 2.79, 2.99, 3.18])
p180c = np.array([3.47, 2.59, 2.72, 3.34, 3.09])
p181c = np.array([3.78, 2.81, 2.85, 2.97, 2.70])
p182c = np.array([2.54, 2.22, 2.24, 2.38, 2.49])
p183c = np.array([3.49, 3.07, 2.98, 3.34, 3.05])
p185c = np.array([1.86, 1.78, 1.65, 2.08, 1.87])
t180c = np.array([3.71, 2.58, 2.83, 2.97, 3.36])
t181c = np.array([2.89, 3.27, 3.23, 3.42, 3.99])
t182c = np.array([2.33, 2.91, 2.80, 2.68, 2.62])
t183c = np.array([1.45, 1.84, 1.65, 1.67, 2.16])
t185c = np.array([2.76, 3.57, 2.55, 4.28, 3.85])


h180cm = np.mean(h180c)
h181cm = np.mean(h181c)
h182cm = np.mean(h182c)
h183cm = np.mean(h183c)
h185cm = np.mean(h185c)
p180cm = np.mean(p180c)
p181cm = np.mean(p181c)
p182cm = np.mean(p182c)
p183cm = np.mean(p183c)
p185cm = np.mean(p185c)
t180cm = np.mean(t180c)
t181cm = np.mean(t181c)
t182cm = np.mean(t182c)
t183cm = np.mean(t183c)
t185cm = np.mean(t185c)

h180cs = np.std(h180c)
h181cs = np.std(h181c)
h182cs = np.std(h182c)
h183cs = np.std(h183c)
h185cs = np.std(h185c)
p180cs = np.std(p180c)
p181cs = np.std(p181c)
p182cs = np.std(p182c)
p183cs = np.std(p183c)
p185cs = np.std(p185c)
t180cs = np.std(t180c)
t181cs = np.std(t181c)
t182cs = np.std(t182c)
t183cs = np.std(t183c)
t185cs = np.std(t185c)

h18am = np.array([h180am, h181am, h182am, h183am, h185am])/h180am*100
p18am = np.array([p180am, p181am, p182am, p183am, p185am])/p180am*100
t18am = np.array([t180am, t181am, t182am, t183am, t185am])/t180am*100
h18as = np.array([h180as, h181as, h182as, h183as, h185as])/h180am*100
p18as = np.array([p180as, p181as, p182as, p183as, p185as])/p180am*100
t18as = np.array([t180as, t181as, t182as, t183as, t185as])/t180am*100

h18bm = np.array([h180bm, h181bm, h182bm, h183bm, h185bm])/h180bm*100
p18bm = np.array([p180bm, p181bm, p182bm, p183bm, p185bm])/p180bm*100
t18bm = np.array([t180bm, t181bm, t182bm, t183bm, t185bm])/t180bm*100
h18bs = np.array([h180bs, h181bs, h182bs, h183bs, h185bs])/h180bm*100
p18bs = np.array([p180bs, p181bs, p182bs, p183bs, p185bs])/p180bm*100
t18bs = np.array([t180bs, t181bs, t182bs, t183bs, t185bs])/t180bm*100

h18cm = np.array([h180cm, h181cm, h182cm, h183cm, h185cm])/h180cm*100
p18cm = np.array([p180cm, p181cm, p182cm, p183cm, p185cm])/p180cm*100
t18cm = np.array([t180cm, t181cm, t182cm, t183cm, t185cm])/t180cm*100
h18cs = np.array([h180cs, h181cs, h182cs, h183cs, h185cs])/h180cm*100
p18cs = np.array([p180cs, p181cs, p182cs, p183cs, p185cs])/p180cm*100
t18cs = np.array([t180cs, t181cs, t182cs, t183cs, t185cs])/t180cm*100

print(h18am)
print(h18as)
FS = 17
fig, ax = plt.subplots(2,3, figsize=(14,8.5),sharey="all")
ax[0,0].errorbar(D, h18am, yerr=h18as, uplims=True, lolims=True, label="T47D")
ax[0,0].errorbar(D, p18am, yerr=p18as, uplims=True, lolims=True, label="T47D-P")
ax[0,0].errorbar(D, t18am, yerr=t18as, uplims=True, lolims=True, label="T47D-T")
ax[0,0].text(0.05, 0.95, "h18, a, mandag (Henrik)", transform=ax[0,0].transAxes)
ax[0,1].errorbar(D, h18bm, yerr=h18bs, uplims=True, lolims=True, label="T47D")
ax[0,1].errorbar(D, p18bm, yerr=p18bs, uplims=True, lolims=True, label="T47D-P")
ax[0,1].errorbar(D, t18bm, yerr=t18bs, uplims=True, lolims=True, label="T47D-T")
ax[0,1].text(0.05, 0.95, "h18, b, tirsdag (Henrik)", transform=ax[0,1].transAxes)
ax[0,2].errorbar(D, h18cm, yerr=h18cs, uplims=True, lolims=True, label="T47D")
ax[0,2].errorbar(D, p18cm, yerr=p18cs, uplims=True, lolims=True, label="T47D-P")
ax[0,2].errorbar(D, t18cm, yerr=t18cs, uplims=True, lolims=True, label="T47D-T")
ax[0,2].text(0.05, 0.95, "h18, c, mandag (Jenny)", transform=ax[0,2].transAxes)
for i in range(2):
    for j in range(3):
        ax[i,j].legend(fontsize=FS-2)
        ax[i,j].set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
        ax[i,j].set_xlabel("Dosis (Gy)",fontsize=FS)
        ax[i,j].tick_params(axis='both', which='major', labelsize=FS)
        ax[i,0].set_ylabel("Mitotisk index (%)",fontsize=FS)
plt.tight_layout()
plt.savefig("tgfbeta_mitoticindex.pdf")
plt.show()
