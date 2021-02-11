import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cycler
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import matplotlib.image as mpimg


colors = cycler('color',
                ['#EE6666', '#3388BB', '#9988DD',
                 '#EECC55', '#88BB44', '#FFBBBB'])
plt.rc('grid', color='gray', linestyle='solid', linewidth=4)
plt.rc('axes', facecolor='w', edgecolor='none', axisbelow=True, grid=True, prop_cycle=colors)
plt.rc('xtick', direction='out', color='gray')
plt.rc('ytick', direction='out', color='gray')
plt.rc('patch', edgecolor='#E6E6E6')
plt.rc('lines', linewidth=2)


FS = 14 # fontsixe til superoverskrift
fs = FS - 2  # fontsize til lengend()
FFS = FS + 3
fig, axq = plt.subplots()

xx = 5
yy = 6
#
# d = mpimg.imread('rect10.png')
# imagebox = OffsetImage(d, zoom=0.2)
# ab = AnnotationBbox(imagebox, (2, 3))
# ax.add_artist(ab)

x = 1
y = 2

r = 0.105   # radius
dr = 0.01
l = 0.8     # længde på skaftet
ll = r

colour = 'firebrick'

# ioniseringskammerets runde tup
circle = plt.Circle((x, y+ll), radius=r, fc=colour)
plt.gca().add_patch(circle)

# ioniseringskammerets skaft fra stregen op til rundingen starter
ax = x+r
ay = y-l
bx = x-r
by = y-l
cx = x-r
cy = y
dx = x+r
dy = y
points = [[ax, ay], [bx, by],[cx,cy], [dx, dy] ]
rect = plt.Polygon(points, fc=colour)
plt.gca().add_line(rect)

# den lange del af skaftet
aax = dx
aay = dy
bbx = cx
bby = cy
ccx = bx
ccy = cy+ll
ddx = dx
ddy = cy+ll
apoints = [[aax, aay], [bbx, bby],[ccx,ccy], [ddx, ddy] ]
arect = plt.Polygon(apoints, fc=colour)
plt.gca().add_line(arect)



xs = 0.3
ys = 0.5
w = 0.15

# tape som holder ioniseringskammerets på plads på pladen
tax = ax + xs
tay = ay + ys
tbx = bx - xs
tby = by + ys
tcx = cx - xs
tcy = cy -  w
tdx = dx + xs
tdy = dy -  w
tpoints = [[tax, tay], [tbx, tby],[tcx,tcy], [tdx, tdy] ]
trect = plt.Polygon(tpoints, fc='khaki')
plt.gca().add_line(trect)

b=2
h=1.1
# stregen som angiver det følsomme punkt på ioniseringskammeret
lax = x+(r-dr/b)
lay = y-dr*h
lbx = x-(r-dr/b)
lby = y-dr*h
lcx = x-(r-dr/b)
lcy = y+dr*h
ldx = x+(r-dr/b)
ldy = y+dr*h
lpoints = [ [lax, lay], [lbx, lby], [lcx,lcy], [ldx, ldy] ]
lrect = plt.Polygon(lpoints, fc='black')
plt.gca().add_line(lrect)

# ledningen
plt.axvline(x=x, ymin=0, ymax=0.28, color='midnightblue')

# gitteret på perspexpladen
grid = np.zeros( (yy,xx) )
axq.imshow(grid, cmap=plt.cm.Blues, vmin=grid.min(), vmax=grid.max())
axq.invert_yaxis()
axq.tick_params(axis='both', which='major', labelsize=fs)
axq.set_xticks([0,1,2,3,4])
plt.tight_layout()
plt.draw()
plt.savefig("gitter.pdf")
plt.show()
