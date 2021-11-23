"""
@author: Henrik Vorup Mortensen
Python Version: 3.9

Formål:
Vi måler antal ioniseringer i gitteret på perspex-pladen som ligger i
røntgenapparatets eksponeringskammer med et dosimeter som fæstes med tape i
hvert af de 30 midterste punkter.
Målingerne gives i nC (nanoCoulomb).

Udregner og plotter gennemsnitsdosis, standardafvig, totalgennemsnitsdosis,
standard error og standard error of the mean.
Udregner afviget i den i'ende måling i hvert punktmålingssæt.

Standardafvig i procent.

Dosen udregnes ved at gange målingen med en kalibreringsfaktor:
K = 43.77 ± 0.039 mGy/nC (milliGray/nanoCoulomb).
Korrektionsfaktorer tilføjes til kalibreringsfaktoren.
NB! Dosis udregnes ikke, intensitetsfelt i procent!

Forsøget gentages n gange.
Gennemsnitsdosen findes ved at tage gennemsnittet af gennemsnitsdoserne fra
hvert forsøg med numpy.mean().
Dosens standardafvig (SD) findes med numpy.std() og fejlen (SEM = SD/sqrt(n))
findes med numpy.std()/sqrt(n).

Begrænsninger:
For at numpy.loadtxt() skal kunne læse txt.-filen må der ikke være forskel i
rækkernes længde.
Det betyder, at forsøget altid skal laves med samme antal målinger i hvert
punkt,hvis numpy.loadtxt() skal kunne klare at læse filen.
"""
################################################################################
################################################################################
################################################################################
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cycler
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox, AnchoredOffsetbox
import matplotlib.image as mpimg
import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'cm'
mpl.rcParams['font.sans-serif'] = 'Times New Roman'
mpl.rcParams['font.family'] = 'Times New Roman'
################################################################################



################################################################################
#### VÆLG TXT.-FIL SOM SKAL KØRES ####
filename = 'xray_data_SSD50_20s.txt'; t = 20; SSD = "SSD50"
# filename = 'xray_data_SSD40_14s.txt'; t = 14; SSD = "SSD40"
filename = 'xray_data_SSD40_13s.txt'; t = 13; SSD = "SSD40"
# filename = 'xray_data_SSD37_5_12s.txt'; t = 12; SSD = "SSD37_5"
################################################################################



################################################################################
#### KONSTANTER ####
#omregningsfaktor K for dosimeteret i mGy/nC (milliGray pr nanoCoulomb)
#faktorer findes i IAEA-rapporten TRS 227 (Andreo et al. 1987)
ku = 1                              #corrects for the change in response as the spectral distribution changes when moving from air to water
muen_over_rho = 1.075               #ratio of the mass energy absorption coefficient between water and air, averaged over the photon spectrum at reference depth
pu = 1.02                           #perturbation factor
kTP = 1.018                         #corrects for the change in temperature and pressure from the calibration to the measurement
NK = 43.77                          #air kerma calibration factor for the given beam quality
K = NK*ku*muen_over_rho*pu*kTP
DNK = 0.39
DC = 0.02

print("t =", t, "sekunder")
# sortér data i kolonner
data = np.loadtxt(filename)
f = data[0:,0]      # første kolonne = eksperiment nummer
y_ = data[0:,1]     # anden kolonne = y-kordinaten
x_ = data[0:,2]     # tredje kolonne = x-koordinaten
C = data[:,3:]      # fjerde til sidste kolonne = aflæst CHRG (nC)
D = data[:,3:]#*K      # CHRG --> Dose ... nC --> Gy (dropper omregning til Gy siden strålekort er i procent)

# nogle praktiske størrelser som bruges i loops
x = int(x_.max() + 1)   # antal målepunkter pr linje på perspexpladen = 5
y = int(y_.max() + 1)   # antal linjer med 5 målepunkter på perspexpladen = 6
N = y*x                 # antal målingspunkter på perspex-pladen = 30 (kan også være flere)
n = int(f.max() + 1)    # antal forsøg gjort ( = 3 hvis færdig)
m = C.shape[1]          # antal målinger pr punkt
################################################################################



################################################################################
#### UDREGN FEJL I DOSEN: (∆D/D)^2 = (∆C/C)**2 + (∆NK/NK)**2 I HVERT MÅLEPUNKT ####
# Tager højde for målingsusikkerhed på ± 0.02 nC og ionkammerusikkerhed på ± 0.39
DD = np.zeros_like(D)   # fejlmatricen ∆D har samme form som dosismatricen D
for i in range(N*n):
    for j in range(m):
        DD[i,j] = np.sqrt( (DC/C[i,j])**2 + (DNK/NK)**2 )*D[i,j]
################################################################################



################################################################################
#### UDREGN GENNEMSNITTET AF MÅLINGERNE I HVERT PUNKT ####
D_Matrix = np.zeros((n, y, x))                # 3 6x5-matricer med hvert enkelt forsøgs dosisgnsn i hver sin matrice
S_Matrix = np.zeros((n, y, x))                # 3 6x5-matricer med hvert enkelt forsøgs SD i hver sin matrice
PS_Matrix = np.zeros((n, y, x))               # SD i i procent
PD_Matrix = np.zeros((n, y, x))               # dosis i procent

d_matrix = np.zeros((y, x))                   # 6x5-matrice: dosisgnsn som rækker
s_matrix = np.zeros((y, x))                   # 6x5-matrice: SD som rækker
yd = np.zeros(x)                                # 5-array: dosisgnsn for hvert punkt på en y-linje
ys = np.zeros(x)                                # 5-array: SD for hvert punkt på en y-linje
for i in range(n):                              # gå igennem alle forsøg
    for j in range(y):                          # gå igennem alle rækker
        for k in range(x):                      # gå igennem alle målepunkter
            yd[k] = np.mean(D[k + j*x + i*N])   # gnsn af hvert punktsæt
            ys[k] = np.std(D[k + j*x + i*N])    # SD af hvert punktsæt
        d_matrix[j] = yd                        # 6x5-matrice med gnsn
        s_matrix[j] = ys                        # 6x5-matrice med SD
    D_Matrix[i] = d_matrix                      # 3 6x5 matricer med gnsn
    S_Matrix[i] = s_matrix                      # 3 6x5 matricer med SD
    PD_Matrix[i] = D_Matrix[i]/D_Matrix.max()*100 # dosis i procent / skaleret på maksdosen - PLOT
    PS_Matrix[i] = S_Matrix[i]/D_Matrix[i]*100  # SD i procent - PLOT

#### MATRICER MED GENNEMSNITSDOSIS, SE, OG SEM I HVERT MÅLEPUNKT
Mean_Of_Total = np.zeros(N)
Mean_Of_Means = np.zeros(N)
SD = np.zeros(N)
SE = np.zeros(N)
SEM = np.zeros(N)
SDtot = np.zeros(N)
SEtot = np.zeros(N)
SEMtot = np.zeros(N)
AllPoints = np.zeros((n, m))                          # for hvert målepunkt; lav en 3-array med hver alle enkeltmålinger D i samme punkt
AllErrors = np.zeros((n, m))                          # for hvert målepunkt; lav en 3-array med hver alle enkeltfejl ∆D i samme punkt
PointMeans = np.zeros(n)                              # for hvert målepunkt; lav en 3-array med hver alle gennemsnitsmålinger <D> i samme punkt
PointErrors = np.zeros(n)                             # for hvert målepunkt; lav en 3-array med hver alle gennemsnitsfejl <∆D> i samme punkt
for i in range(N):                                    # gå igennem alle målepunkter på perspexpladen: range(N) = 0,1,2,...29
    for j in range(n):                                # gå igennem alle målinger i hvert punkt: range(n) = 0,1,2
        AllPoints[j] = D[i + N*j]                     # alle enkeltmålinger fra alle forsøg
        AllErrors[j] = DD[i + N*j]                    # alle enkeltmålingers standardfejl fra alle forsøg
        PointMeans[j] = np.mean(D[i + N*j])           # gnsn af alle punktmålinger fra hvert forsøg
        PointErrors[j] = np.mean(DD[i + N*j])         # gnsn af alle punkters fejl fra hvert forsøg
    Mean_Of_Total[i] = np.mean(AllPoints)             # N-matrice med gnsn af alle målinger i hvert målepunkt
    Mean_Of_Means[i] = np.mean(PointMeans)            # N-matrice med gnsn af gnsn i hvert målepunkt (skal give samme som ovenstående)
    SD[i] = np.std(AllPoints)                         # SD i hvert målepunkt
    SE[i] = np.std(AllPoints)/np.sqrt(n*m)            # SE i hvert målepunkt
    SEM[i] = np.std(PointMeans)/np.sqrt(n)            # SEM i hvert målepunkt
    SDtot[i] = np.sqrt( (np.std(AllPoints)**2 + np.mean(AllErrors)**2 ))   # SD + fejl DD i hvert målepunkt
    SEtot[i] = np.sqrt( (np.std(AllPoints)/np.sqrt(n*m))**2 + np.mean(AllErrors)**2 )# SE + fejl DD i hvert målepunkt
    SEMtot[i] = np.sqrt( (np.std(PointMeans)/np.sqrt(n))**2 + np.mean(PointErrors)**2 )# SEM + fejl DD i hvert målepunkt - PLOT

#### MATRICER TIL TOTALGENNEMSNIT
ya = np.zeros(x)
ym = np.zeros(x)

ysd = np.zeros(x)
yse = np.zeros(x)
ysem = np.zeros(x)

ysdtot = np.zeros(x)
ysetot = np.zeros(x)
ysemtot = np.zeros(x)

DA_Matrix = np.zeros((y, x))          # matrice til totalgennemsnit udregnet fra samtlige n*m målinger i hvert punkt
DM_Matrix = np.zeros((y, x))          # matrice til totalgennemsnit udregnet fra n gennemsnit i hvert punkt
PDM_Matrix = np.zeros((y, x))          # matrice til totalgennemsnit udregnet fra n gennemsnit i hvert punkt

SD_Matrix = np.zeros((y, x))          # matrice til SE i hvert punkt
SE_Matrix = np.zeros((y, x))          # matrice til SE i hvert punkt
SEM_Matrix = np.zeros((y, x))         # matrice til SEM i hvert punkt

SDtot_Matrix = np.zeros((y, x))         # matrice til SD + fejl DD i hvert punkt
SEtot_Matrix = np.zeros((y, x))         # matrice til SE + fejl DD i hvert punkt
SEMtot_Matrix = np.zeros((y, x))        # matrice til SEM + fejl DD i hvert punkt

PSD_Matrix = np.zeros((y, x))           # SD i procent
PSE_Matrix = np.zeros((y, x))           # SE i procent
PSEM_Matrix = np.zeros((y, x))          # SEM i procent

PSDtot_Matrix = np.zeros((y, x))        # totalfejl med SD + fejl DD i procent
PSEtot_Matrix = np.zeros((y, x))        # totalfejl med SE + fejl DD i procent
PSEMtot_Matrix = np.zeros((y, x))       # totalfejl med SEM + fejl DD i procent

for i in range(y):                      # gå igennem hver linje på perspexpladen
    for j in range(x):                  # gå igennem hvert punkt på perspexpladen
        ya[j] = Mean_Of_Total[j + i*x]  # fyld 6 5-arrays (én for hver linje på pladen) med gnsn af alle målinger
        ym[j] = Mean_Of_Means[j + i*x]  # fyld 6 5-arrays (én for hver linje på pladen) med gnsn af gennemsnittene fra hver forsøg
        ysd[j]  = SD[j + i*x]           # SD
        yse[j]  = SE[j + i*x]           # fyld 6 5-arrays (én for hver linje på pladen) med standard error (SE)
        ysem[j] = SEM[j + i*x]          # fyld 6 5-arrays (én for hver linje på pladen) med standard error of the mean (SEM)
        ysdtot[j]  = SDtot[j + i*x]     # fyld 6 5-arrays (én for hver linje på pladen) med standard error (SD)
        ysetot[j]  = SEtot[j + i*x]     # fyld 6 5-arrays (én for hver linje på pladen) med standard error (SE)
        ysemtot[j] = SEMtot[j + i*x]    # fyld 6 5-arrays (én for hver linje på pladen) med standard error of the mean (SEM)
    DA_Matrix[i] = ya                   # fyld 6x5-dosematrix til plotting
    DM_Matrix[i] = ym                   # fyld 6x5-dosematrix til plotting

    PDM_Matrix[i] = DA_Matrix[i]/D_Matrix.max()*100

    SD_Matrix[i] = ysd                  # fyld 6x5-SD-matrix til plotting
    SE_Matrix[i] = yse                  # fyld 6x5-SE-matrix til plotting
    SEM_Matrix[i] = ysem                # fyld 6x5-SEM-matrix til plotting

    SDtot_Matrix[i] = ysdtot                  # fyld 6x5-SE-matrix til plotting
    SEtot_Matrix[i] = ysetot                  # fyld 6x5-SE-matrix til plotting
    SEMtot_Matrix[i] = ysemtot                # fyld 6x5-SEM-matrix til plotting

    PSD_Matrix[i] = SD_Matrix[i]/DM_Matrix[i]*100
    PSE_Matrix[i] = SE_Matrix[i]/DM_Matrix[i]*100
    PSEM_Matrix[i] = SEM_Matrix[i]/DM_Matrix[i]*100

    PSDtot_Matrix[i] = SDtot_Matrix[i]/DM_Matrix[i]*100
    PSEtot_Matrix[i] = SEtot_Matrix[i]/DM_Matrix[i]*100
    PSEMtot_Matrix[i] = SEMtot_Matrix[i]/DM_Matrix[i]*100
################################################################################



################################################################################
#### Sikkerhed for at SE og SEM er rigtig udregnet ####
if DA_Matrix.any() != DM_Matrix.any():
    print('ADVARSEL!! Gennemssnitsdose udregnet fra alle målinger ≠ gennemsnitsdose udregnet fra gennemsnittene')
################################################################################



################################################################################
#### STRÅLINGSINTENSITETPLOT ####
# intesitetsregulering i plottene så de bliver ensartede og sammenlignbare
# faste intensitetsværdier på alle fejlplot så de kan sammenlignes på tværs af forsøg
D_min = D_Matrix.min()
D_max = D_Matrix.max()
S_min = 1 #S_Matrix.min()
S_max = 3.3 #S_Matrix.max()
P_min = 1 #S_Matrix.min()
P_max = 4.5 #S_Matrix.max()

DM_min = D_min
DM_max = D_max
SE_min = S_min  # SE_Matrix.min()
SE_max = S_max  # SE_Matrix.max()
SEM_min = S_min # SEM_Matrix.min()
SEM_max = S_max # SEM_Matrix.max()

FS = 14+3 # fontsize
fs = FS - 2  # fontsize til lengend()
FFS = FS + 3
################################################################################



################################################################################
### STRÅLEFELTKARAKTERISTIK
titlea = "%s, %ss: %s målinger pr punkt"
if n == 1:
    figa, axa = plt.subplots(1,3,figsize=(10, 3.9))
    axa[0].imshow(D_Matrix[0], vmin=D_min, vmax=D_max, cmap='inferno',interpolation='lanczos')#, interpolation='lanczos')
    axa[0].set_title("Dosis: $\\overline{D}$ ± SD (%)",fontsize=FS)
    axa[0].invert_yaxis()
    axa[0].tick_params(axis='both', which='major', labelsize=fs)
    axa[0].set_xticks([0,1,2,3,4])
    axa[1].imshow(PS_Matrix[0], vmin=P_min, vmax=P_max, cmap=plt.cm.Blues,interpolation='lanczos')#, interpolation='lanczos')
    axa[1].set_title("Standardafvig SD (%)",fontsize=FS)
    axa[1].invert_yaxis()
    axa[1].tick_params(axis='both', which='major', labelsize=fs)
    axa[2].set_xticks([0,1,2,3,4])
    axa[2].imshow(PSDtot_Matrix, vmin=P_min, vmax=P_max, cmap=plt.cm.Blues,interpolation='lanczos')#, interpolation='lanczos')
    axa[2].set_title("Total fejl i dosen (%)",fontsize=FS)#,\n $\Delta D_{SD}$ = $[$SD$^2$ + $\\langle\Delta D\\rangle]^{1/2}$  ",fontsize=FS)
    axa[2].invert_yaxis()
    axa[2].tick_params(axis='both', which='major', labelsize=fs)
    axa[2].set_xticks([0,1,2,3,4])
    for i in range(x):
        for j in range(y):
            c = PD_Matrix[0,j,i]
            d = PS_Matrix[0,j,i]
            e = PS_Matrix[0,j,i]
            f = PSDtot_Matrix[j,i]
            axa[0].text(i, j, '%.1f' %c, va='center', ha='center', fontsize=fs)
            axa[1].text(i, j,    '%.1f' %e, va='center', ha='center', fontsize=fs)
            axa[2].text(i, j,    '%.1f' %f, va='center', ha='center', fontsize=fs)
    plt.tight_layout()
else:
    figa, axa = plt.subplots(nrows=2,ncols=n,figsize=(10, 8))
    for i in range(n):
        axa[0,i].imshow(D_Matrix[i], vmin=D_min, vmax=D_max, cmap='inferno',interpolation='lanczos')
        axa[0,i].set_title("Dosis $\\overline{D}$ (%%), %s. forsøg"%(int(i+1)),fontsize=FS )
        axa[0,i].invert_yaxis()
        axa[0,i].tick_params(axis='both', which='major', labelsize=fs)
        axa[0,i].set_xticks([0,1,2,3,4])
        axa[1,i].imshow(PS_Matrix[i], vmin=P_min, vmax=P_max, cmap=plt.cm.Blues,interpolation='lanczos')
        axa[1,i].set_title("Standardafvig SD (%)",fontsize=FS)
        axa[1,i].invert_yaxis()
        axa[1,i].tick_params(axis='both', which='major', labelsize=fs)
        axa[1,i].set_xticks([0,1,2,3,4])
        for j in range(x):
            for k in range(y):
                c = PD_Matrix[i,k,j]
                d = PS_Matrix[i,k,j]
                e = PS_Matrix[i,k,j]
                axa[0,i].text(j, k,    '%.1f' %c, va='center', ha='center', fontsize=fs)
                axa[1,i].text(j, k,    '%.1f' %e, va='center', ha='center', fontsize=fs)
    plt.tight_layout()
plt.savefig("%s_%s_dosiogSD.pdf"%(SSD,t))
################################################################################



################################################################################
#plot gennemsnitsdose og SEM fra alle forsøg
titleb = "%s, %ss: %s forsøg, %s målinger pr punkt "
figb, axb = plt.subplots(nrows=1,ncols=3,figsize=(10, 3.9))
axb[0].imshow(DM_Matrix, vmin=DM_min, vmax=DM_max, cmap='inferno',interpolation='lanczos')
axb[0].set_title("Dosis $\\langle D\\rangle$ (%)",fontsize=FS)
axb[0].invert_yaxis()
axb[0].tick_params(axis='both', which='major', labelsize=fs)
axb[0].set_xticks([0,1,2,3,4])
axb[1].imshow(PSEM_Matrix, vmin=P_min, vmax=P_max, cmap=plt.cm.Greens,interpolation='lanczos')
axb[1].set_title("SEM (%)",fontsize=FS)#Standard error of the mean (%),\n SEM = ${ \\frac{ SD(\overline{D}) }{ \\sqrt{n} } } $  ",fontsize=FS)
axb[1].invert_yaxis()
axb[1].tick_params(axis='both', which='major', labelsize=fs)
axb[1].set_xticks([0,1,2,3,4])
axb[2].imshow(PSEMtot_Matrix, vmin=P_min, vmax=P_max, cmap=plt.cm.Blues,interpolation='lanczos')
axb[2].set_title("Total fejl i dosen (%)",fontsize=FS)#,\n $\Delta D_{SEM}$ = $[$SEM$^2$ + $\\langle\Delta D\\rangle]^{1/2}$  ",fontsize=FS)
axb[2].invert_yaxis()
axb[2].tick_params(axis='both', which='major', labelsize=fs)
axb[2].set_xticks([0,1,2,3,4])
for i in range(x):
    for j in range(y):
        b = PSEM_Matrix[j,i]
        c = PDM_Matrix[j,i]
        axb[0].text(i, j, '%.1f' %c, va='center', ha='center', fontsize=fs)
        d = PSEM_Matrix[j,i]
        e = PSEMtot_Matrix[j,i]
        g = np.array([d, e])
        for k in range(2):
            axb[k+1].text(i, j, '%.1f' %(g[k]), va='center', ha='center', fontsize=fs)
plt.tight_layout()
plt.savefig("%s_%s_gnsnSEogSEM.pdf"%(SSD,t))
################################################################################



################################################################################
### NORMALISEREDE 1.-MÅLINGER

#### NORMALISERING AF ALLE MÅLINGSINDEKS
#### FOR AT SE HVOR MEGET MERE RØRET GIVER
#### I DEN FØRSTE MÅLING
norm = np.zeros_like(D)             # norm
NormArray = np.zeros((m, n*N))    # NormArray
# norm_wi = np.zeros_like(D)          # norm without index i
# norm_wi0 = np.zeros_like(D)          # norm without index i
# NormArray_wi = np.zeros((m, n*N)) # NormArray without index i
# NormArray_wi0 = np.zeros((m, n*N)) # NormArray without index 0

# udregn normaliserede værdier og fyld matricer til PLOTTING
for i in range(n*N):                                                        # hele datasættet igennem (90 punkter)
    norm[i] = D[i]/np.mean(D[i])                                            # normaliser alle målinger ift eget punktsæt
    # norm_wi0[i] = D[i]/np.mean(D[i,1:])
    for j in range(m):                                                      # 5 eller 10 gange, dvs en gang pr målingsindeks
        NormArray[j,i] = norm[i,j]                                          # array med til plotting
        # norm_wi[i] = D[i]/np.mean(np.concatenate((D[i,:j], D[i,j+1:])))     # normaliser måling ift resterende målinger (ikke egen måling)
        # NormArray_wi[j,i] = norm_wi[i,j]                                    # array med til plotting
        # NormArray_wi0[j,i] = norm_wi0[i,j]


PN = np.zeros((n, y, x))
for i in range(n):
    for j in range(y):
        for k in range(x):
            PN[i,j,k] = (NormArray[i,j*x + k]-1)*100


################################################################################
### Kort over hvor på pladen førstemålingen afviger mest
if n == 1: #hvis kun ét forsøg
    figx, axx = plt.subplots(1,1)
    axx.imshow(PN[0], cmap='hsv')
    axx.set_title("Afvig (%)",fontsize=FS)
    axx.invert_yaxis()
    axx.tick_params(axis='both', which='major', labelsize=fs)
    axx.set_xticks([0,1,2,3,4])
    for i in range(y):
        for j in range(x):
            c = PN[0,i,j]
            axx.text(j,i, '%.1f' %c, va='center', ha='center', fontsize=fs)
    plt.tight_layout()
else: #hvis flere forsøg
    figx, axx = plt.subplots(nrows=1,ncols=n,figsize=(10, 4.4))
    for i in range(n):
        axx[i].imshow(PN[i],cmap='hsv')#, vmin=D_min, vmax=D_max, cmap='inferno',interpolation='lanczos')
        axx[i].set_title("Afvig (%%), %s. forsøg"%(int(i+1)),fontsize=FS )
        axx[i].invert_yaxis()
        axx[i].tick_params(axis='both', which='major', labelsize=fs)
        axx[i].set_xticks([0,1,2,3,4])
        for j in range(x):
            for k in range(y):
                c = PN[i,k,j]
                axx[i].text(j, k,    '%.1f' %c, va='center', ha='center', fontsize=fs)
    plt.tight_layout()
plt.show()
################################################################################



################################################################################
#### PLOT FOR AT SE HVOR MEGET ENKELTE MÅLINGER AFVIGER FRA GENNEMSNITTET MÅLINGER I SAMME KOHORT
# plt.rc('grid', color='w', linestyle='solid')
titlec = " %s, %ss: Alle i'ende-målinger i samme plot normaliseret i forhold til gennemsnittet af punktsættet"
if m <= 5: #hvis kun ét forsøg
    ylimits = (-7, 7)
    figc, axc = plt.subplots(nrows=1,ncols=5,figsize=(14, 3.8), sharey="all")
    for i in range(m):
        avg = np.mean((NormArray[i]-1)*100)
        sem = np.std(NormArray[i]*100)/np.sqrt(len(NormArray[i]))
        axc[i].axhline(y=avg,label="$%.1f\pm%.1f$"%(avg,sem), color="red")
        axc[i].axhline(y=0, linestyle='dotted', color="lightgray")
        axc[i].plot((NormArray[i][:30]-1)*100, ".")#, ".", label="norm. måling")
        axc[i].plot((NormArray[i][30:60]-1)*100, ".")
        axc[i].plot((NormArray[i][60:]-1)*100, ".")
        axc[i].set_title("%s.-målinger"%(i+1),fontsize=FFS)
        axc[i].legend(loc=3, fontsize=fs+1)
        axc[i].set_ylim((ylimits))
        axc[i].set_xlabel("Målepunkt",fontsize=FFS)
        axc[0].set_ylabel("Afvig fra gnsn (%)",fontsize=FFS)
        axc[i].tick_params(axis='both', which='major', labelsize=FFS-1)
        a=axc[i].get_xticks().tolist()
        axc[i].set_xticks(np.array([0, 9, 19, 29]))
        a = ["$(0.0)$", "$(1,3)$", "$(3,1)$", "$(4,5)$"]
        axc[i].set_xticklabels(a)
else: #hvis flere forsøg
    ylimits = (-9, 9)
    figc, axc = plt.subplots(nrows=2,ncols=5,figsize=(14, 7.5), sharey="all")
    for i in range(2):
        for j in range(5):
            avg = np.mean(NormArray[j + i*5]-1)*100
            sem = np.std(NormArray[j + i*5]*100)/np.sqrt(len(NormArray[i]))
            axc[i,j].axhline(y=avg,label="$%.1f\pm%.1f\%%$"%(avg,sem), color="red")
            axc[i,j].axhline(y=0, linestyle='dotted', color="lightgray")
            axc[i,j].plot((NormArray[j + i*5][:30]-1)*100, ".")#, ".", label="norm. måling")
            axc[i,j].plot((NormArray[j + i*5][30:60]-1)*100, ".")
            axc[i,j].plot((NormArray[j + i*5][60:]-1)*100, ".")
            axc[i,j].set_title("%s.-målinger"%(j + 1 + i*5),fontsize=FFS)
            axc[i,j].legend(loc=3, fontsize=fs+1)
            axc[i,j].set_ylim(ylimits)
            axc[i,j].set_xlabel("Målepunkt",fontsize=FFS)
            axc[i,0].set_ylabel("Afvig fra gnsn (%)",fontsize=FFS)
            axc[i,j].tick_params(axis='both', which='major', labelsize=FFS-1)
            a=axc[i,j].get_xticks().tolist()
            axc[i,j].set_xticks(np.array([0, 9, 19, 29]))
            a = ["$(0.0)$", "$(1,3)$", "$(3,1)$", "$(4,5)$"]
            axc[i,j].set_xticklabels(a)
plt.tight_layout()
plt.savefig("%s_%s_norm_rest_alle.pdf"%(SSD,t))
################################################################################



################################################################################
# titled = "%s, %ss: Første måling i hvert punktsæt\n normaliseret i forhold til\n gennemsnittet af punktsættet"
# figd, axd = plt.subplots()
# # axd.set_title(titled%(SSD, t), fontsize=FS)
# axd.plot(NormArray[0], ".", label="normaliseret måling")
# axd.set_ylim(ylimits)
# axd.set_xlabel("Målingsindeks",fontsize = FS)
# axd.set_ylabel("Normaliseret måling og gennemsnit",fontsize = FS)
# avg0 = np.mean(NormArray[0])
# sem0 = np.std(NormArray[0])/np.sqrt(len(NormArray[0]))
# # axd.fill_between(np.array(len(NormArray_wi[0])),avg_wi0-sem_wi0, avg_wi0+sem_wi0)
# axd.axhline(y=avg0,label="gennemsnit: %.3f±%.3f"%(avg0, sem0), color="red")
# axd.axhline(y=1, linestyle='dotted', color="lightgray")
# axd.legend(fontsize = fs)
# axd.tick_params(axis='both', which='major', labelsize=FS)
#
# a=axd.get_xticks().tolist()
# if n == 1:
#     axd.set_xticks(np.array([0,10, 20, 29]))
#     a = [1,11,21,30]
# else:
#     axd.set_xticks(np.array([0,30,60,89]))  #np.linspace(0,int(n*N-1),2))
#     a = [1,31,61,90]
# axd.set_xticklabels(a)
# plt.yticks(fontsize = FS)
# plt.tight_layout()
# # plt.tight_layout(rect=titlecorrection)
# plt.savefig("%s_%s_norm_rest.pdf"%(SSD,t))
# # plt.show()
################################################################################


################################################################################
### PLOT OVER FORSKELLIGE EKSPONERINGSTIDER
### KRÆVER PRINT FRA HVER ENKELT TEXTFIL SOM FYLDES MANUELT IND I ARRAYS
#### LILLE PLOT MED KALIBRERINGSFAKTOR FOR FORSKELLIGE EKSPONERINGSTIDER ####
scope = np.array([12, 13, 14, 20])

avg0 = np.mean(NormArray[0])
std0 = np.std(NormArray[0])
sem0 = np.std(NormArray[0])/np.sqrt(len(NormArray[0]))
# print(avg0)
# print(std0)
# print(sem0)
first_meas_norms = np.array([1.0215465117895153, 1.0204540638976112, 1.0120538668226677, 1.0087396484416213 ])
first_meas_stds = np.array([0.0353466377428203, 0.02763525154902737, 0.026603320855124565,0.016202058486188126])
first_meas_sems = np.array([0.006453383607902055, 0.0029130112868874284, 0.002804236240881732, 0.0017078469199871408 ])

####Plot
# titlee = "Kalibreringsfaktor $\\delta(t)$ for kort eksponering"
# fige, axe = plt.subplots()
# # axe.set_title(titlee, fontsize=FS)
# qqq = (first_meas_norms-1)*100
# ppp = (first_meas_sems)*100
# axe.plot(scope,qqq, "r", label="1.-målingens gennemsnitlige afvig")
# axe.fill_between(scope, qqq-ppp, qqq+ppp, alpha=0.3,label="SEM")
# axe.set_xlabel("Eksponeringstid (s)",fontsize=FS)
# axe.set_ylabel("Afvig fra gennemsnittet (%)",fontsize=FS)
# axe.legend(fontsize = FS)
# # axe.set_yticks([1.006, 1.008, 1.010, 1.012, 1.014, 1.016, 1.018, 1.020, 1.022, 1.024, 1.026, 1.028])
# plt.xticks(fontsize=FS)
# plt.yticks(fontsize=FS)
# plt.tight_layout()
# # plt.tight_layout(rect=titlecorrection)
# plt.savefig("Kalib_faktor_over_tid.pdf")
# plt.show()
################################################################################



################################################################################
#### KALIBREINGSFAKTOR MED RENSEDE NORMALISERINGER SÅ FØRSTEINDEKSEN IKEN MEDGÅR I GENNEMSNITTET ####
# IKKE BRUGT
# avg0_wi0 = np.mean(NormArray_wi0[0])
# std0_wi0 = np.std(NormArray_wi0[0])
# sem0_wi0 = np.std(NormArray_wi0[0])/np.sqrt(len(NormArray_wi0[0]))
# print(avg0_wi0)
# print(std0_wi0)
# print(sem0_wi0)
# first_meas_norms_wi0 = np.array([1.0241531756700784, 1.0228734481098294, 1.0153356249882308, 1.0110310657488206])
# first_meas_stds_wi0 = np.array([0.039411262667292044, 0.030849849067484964, 0.033392861579865804, 0.020344157007350143])
# first_meas_sems_wi0 = np.array([0.007195479194212357,0.003251859617522467, 0.0035199166727701535, 0.0021444624406433783])

# titleg = "Kalibreringsfaktor $\\delta(t)$ for kort eksponering uden førstemåling"
# figg, axg = plt.subplots()
# # axg.set_title(titlee, fontsize=FS)
# axg.plot(scope,first_meas_norms_wi0, "m", label="$\\delta(t)$ (uden indeks 1)")
# axg.fill_between(scope, first_meas_norms_wi0-first_meas_sems_wi0, first_meas_norms_wi0+first_meas_sems_wi0,alpha=0.3,label="SEM")
# axg.plot(scope,first_meas_norms, "r--", label="$\\delta(t)$")
# axg.fill_between(scope, first_meas_norms-first_meas_sems, first_meas_norms+first_meas_sems,alpha=0.3,label="SEM")
# axg.set_xlabel("Eksponeringstid (s)",fontsize=FS)
# axg.set_ylabel("Kalibreringsfaktor $\\delta(t)$",fontsize=FS)
# axg.legend(fontsize = FS)
# # axg.set_yticks([1.006, 1.008, 1.010, 1.012, 1.014, 1.016, 1.018, 1.020, 1.022, 1.024, 1.026, 1.028])
# plt.xticks(fontsize=FS)
# plt.yticks(fontsize=FS)
# plt.tight_layout()
# # plt.tight_layout(rect=titlecorrection)
# plt.savefig("Kalib_faktor_over_tid_renset.pdf")
################################################################################



################################################################################
#### GENNEMSNITLIG STANDARDAFVIG I DET HOMOGENE OMRÅDE AF EKSPONERINGSREGIONEN ####
P_Hom_mean = np.mean(PS_Matrix[:,2:-1,1:-1])
P_Hom_std = np.std(PS_Matrix[:,2:-1,1:-1])
P_Hom_sem = np.std(PS_Matrix[:,2:-1,1:-1])/np.sqrt(PS_Matrix[:,2:-1,1:-1].shape[0]*PS_Matrix[:,2:-1,1:-1].shape[1])
# print((PS_Matrix[:,2:-1,1:-1]))
# print( PS_Matrix[:,2:-1,1:-1].shape[0]*PS_Matrix[:,2:-1,1:-1].shape[1])
# print("P_Hom_mean =", P_Hom_mean)
# print("P_Hom_std =", P_Hom_std)
# print("P_Hom_sem =", P_Hom_sem)
meanstd_hom = np.array([3.0609518709544905, 2.9058740060196473, 2.5327468220836034, 1.6194550102161929])
stdstd_hom = np.array([0.508958148510818, 0.618839073735935, 0.6038044325544453, 0.5369453932178103])
semstd_hom = np.array([0.29384712404897434, 0.20627969124531167, 0.2012681441848151, 0.17898179773927012])
####PLOT
# # titlef = "Gennemsnitligt standardafvig i det homogene felt som funktion af eksponeringstid"
# figf, axf = plt.subplots()
# # axe.set_title(titlee, fontsize=FS)
# im = mpimg.imread('gitter.png')
# axf.imshow(im, extent=[11.5,14.65,1.5,2.26], aspect='auto')
# axf.plot(scope,meanstd_hom, label="$\\langle$SD$\\rangle(t)$")
# axf.fill_between(scope, meanstd_hom - semstd_hom, meanstd_hom + semstd_hom,alpha=0.3,label="SEM")
# axf.set_xlabel("Eksponeringstid (s)",fontsize=FS)
# axf.set_ylabel("Gnsn standardafvig $\\langle$SD$\\rangle(t)$ (%)",fontsize=FS)
# axf.legend(fontsize = FS)
# plt.xticks(fontsize=FS)
# plt.yticks(fontsize=FS)
# plt.tight_layout()
# # plt.tight_layout(rect=titlecorrection)
# plt.savefig("Gnsn_std_afvig.pdf")
################################################################################



################################################################################
#### GENNEMSNITLIG SEM I DEN HOMOGENE REGION
PSEM_Hom_mean = np.mean(PSEM_Matrix[2:-1,1:-1])
PSEM_Hom_std = np.std(PSEM_Matrix[2:-1,1:-1])
PSEM_Hom_sem = np.std(PSEM_Matrix[2:-1,1:-1])/np.sqrt(PS_Matrix[:,2:-1,1:-1].shape[0]*PS_Matrix[:,2:-1,1:-1].shape[1])
# print("PSEM_Hom_mean =", PSEM_Hom_mean)
# print("PSEM_Hom_std =", PSEM_Hom_std)
# print("PSEM_Hom_sem =", PSEM_Hom_sem)
scope_sem = np.array([13,14,20])
meansem_hom = np.array([0.8510225461213561,1.3031104776606717,1.5848983129957606])
semsem_hom = np.array([0.053547249218584086,0.15708604238204218,0.07614570548795169])

####PLOT
# # titlef = "Gennemsnitligt standardfejl i det homogene felt som funktion af eksponeringstid"
# figh, axh = plt.subplots()
# im = mpimg.imread('gitter.png')
# # axh.imshow(im, extent=[17.3,20.3,1.07,1.47], aspect='auto')
# axh.imshow(im, extent=[12.6+5,15.2+5,0.8,1.13], aspect='auto')
# axh.plot(scope_sem,meansem_hom, "g", label="$\\langle$SEM$\\rangle(t)$")
#
# axh.fill_between(scope_sem, meansem_hom - semsem_hom, meansem_hom + semsem_hom,alpha=0.3,label="SEM")
# axh.set_xlabel("Eksponeringstid (s)",fontsize=FS)
# axh.set_ylabel("Gnsn standardfejl $\\langle$SEM$\\rangle(t)$ (%)",fontsize=FS)
# axh.legend(fontsize = FS)
# plt.xticks(fontsize=FS)
# plt.yticks(fontsize=FS)
# plt.tight_layout()
# # plt.tight_layout(rect=titlecorrection)
# plt.savefig("Gnsn_SEM.pdf")
# # plt.show()
################################################################################



################################################################################
#### HVER MÅLINGSINDEKS' GENNEMSNITLIGE AFVIG FRA KOHORTEN
navgs = np.zeros(m)
nsems = np.zeros(m)
for i in range(m):
    navgs[i] = (np.mean(NormArray[i] - 1)*100)
    nsems[i] = np.std(NormArray[i] - 1)*100/np.sqrt(len(NormArray[i]))
if m <= 5: #hvis fem målinger i hvert punkt
    l = np.array([1,2,3,4,5])
else: #hvis ti målinger i hvert punkt
    l = np.array([1,2,3,4,5,6,7,8,9,10])
figi, axi = plt.subplots()
axi.plot(l, navgs, "r*", label="Målingsindeksens gennemsnitlige afvig")
axi.fill_between(l, navgs-nsems, navgs+nsems, alpha=0.3,label="SEM")
axi.axhline(y=0, linestyle='dotted', color="k")
axi.set_xlabel("Målingsindeks",fontsize=FS)
axi.set_ylabel("Afvig fra gennemsnittet (%)",fontsize=FS)
axi.set_xticks(l)
axi.legend(fontsize = FS)
plt.xticks(fontsize=FS)
plt.yticks(fontsize=FS)
plt.tight_layout()
plt.savefig("%s_%s_m_index_gnsnafvig-fra-gnsn.pdf"%(SSD,t))
# plt.show()
################################################################################
