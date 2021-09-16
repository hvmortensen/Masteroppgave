import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cycler
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
import matplotlib as mpl

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'cm'
mpl.rcParams['font.sans-serif'] = 'Times New Roman'
mpl.rcParams['font.family'] = 'Times New Roman'


FS = 14+3 # fontsixe til superoverskrift
fs = FS + 3   # fontsize til lengend()
FFS = FS + 1

o = 2
X = 0

def gauss(x):
    return 1/(2*np.pi)*1/o*np.exp(-((x - X)**2)/(2*o**2) )

edge = 6.2
x = np.linspace(-edge, edge, 1501)

fig, ax = plt.subplots()

#
# # stregen som angiver det følsomme punkt på ioniseringskammeret
# lax = o
# lay = gauss(o)
# lbx = o - 0.2
# lby = gauss(o) - 0.2
# lcx = o + 2
# lcy = gauss(o) + 0.2
# ldx = o
# ldy = gauss(o)
# lpoints = [ [lax, lay], [lbx, lby], [lcx,lcy], [ldx, ldy] ]
# lrect = plt.Polygon(lpoints, fc='black')
# plt.gca().add_line(lrect)
w = 0.37
h = 0.0037

xx = 0.77
yy = 0.05
#
# im = mpimg.imread('rho.png')
# ax.imshow(im, extent=[xx,xx+w, yy, yy+h], aspect='auto')

ax.plot(x, gauss(x), linewidth=2)
ax.axvline(x=0, ymin=0.04, ymax=0.98, color="k")
ax.plot(o, gauss(o), "r.")
ax.plot(0, gauss(o), "r.")
ax.axhline(y=gauss(o), xmax=0.5, xmin=0.65, color="r", linestyle="--", label="$\\sigma$")
ax.tick_params(axis='both', which='major', labelsize=fs)
a=ax.get_xticks().tolist()
ax.set_xticks(np.array([-6,-4,-2, 0, 2, 4, 6]))
a = ["", "", "", "$X$","", "\n $x\\longrightarrow$               "]
ax.set_xticklabels(a)
ax.set_yticklabels([])
# ax.legend(fontsize=fs)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
ax.text(0.8,0.05 ,"$\\sigma$",fontsize=fs)
plt.draw()
plt.tight_layout()
plt.savefig("Gausscurve.pdf")
plt.show()
