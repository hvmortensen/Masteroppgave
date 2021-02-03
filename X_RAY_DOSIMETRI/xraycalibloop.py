"""
@author: Henrik Vorup Mortensen
Python Version: 3.9

Formål:
Vi måler antal ioniseringer i gitteret på perspex-pladen som ligger i
røntgenapparatets eksponeringskammer med et dosimeter som fæstes med tape i
hvert af de 30 midterste punkter.
Målingerne gives i nC (nanoCoulomb).

Eksponeringstid: 20 sekunder.
Afstand fra røntgenkilden: 50 cm (SSD50).
Afstand fra røntgenkilden: 40 cm (SSD40).
Filtrering: 1.52 mm Al + 2.60 mm Cu.
Spænding over røntgenrøret: 220 kV (kiloVolt).
Katodestrøm: 10 mA (milliAmpere).

I hvert punkt tages fem målinger.
Gennemsnit (mean) for de fem målinger udregnes med numpy.mean().
Standardafvig (SD) for de fem målinger udregnes med numpy.std()
Dosen udregnes ved at gange gennemsnittet med en omregningsfaktor som gives af
fabrikatet af dosimeteret på K = 43.77 mGy/nC (milliGray/nanoCoulomb)

Forsøget gentages n gange.
Gennemsnitsdosen findes ved at tage gennemsnittet af gennemsnitsdoserne fra
hvert forsøg med numpy.mean().
Dosens standardafvig (SD) findes med numpy.std() og fejlen (SEM = SD/sqrt(n))
findes med numpy.std()/sqrt(n).


Begrænsninger:
For at numpy.loadtxt() skal kunne læse txt.-filen må der ikke være forskel i
kolonnernes længde.
Det betyder, at forsøget altid skal laves med samme antal målinger i hvert
punkt,hvis numpy.loadtxt() skal kunne klare at læse filen.


Spørgsmål: Hvordan inplementerer jeg fejlen i estimatet for
kalibreringsfaktoren: (∆D/D)2 = (∆C/C)2 + (∆K/K)2

Alle målinger af CHRG er estimeret til ±0.02

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cycler
# colors = cycler('color',
#                 ['#EE6666', '#3388BB', '#9988DD',
#                  '#EECC55', '#88BB44', '#FFBBBB'])
# plt.rc('grid', color='w', linestyle='solid')
# plt.rc('axes', facecolor='#E6E6E6', edgecolor='none',
#        axisbelow=True, grid=True, prop_cycle=colors)
# plt.rc('xtick', direction='out', color='gray')
# plt.rc('ytick', direction='out', color='gray')
# plt.rc('patch', edgecolor='#E6E6E6')
# plt.rc('lines', linewidth=2)

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

#### VÆLG TXT.-FIL SOM SKAL KØRES ####
filename = 'xray_data_SSD50_20s.txt'; t = 20; SSD = "SSD50"
# filename = 'xray_data_SSD40_14s.txt'; t = 14; SSD = "SSD40"
# filename = 'xray_data_SSD40_13s.txt'; t = 13; SSD = "SSD40"
# filename = 'xray_data_SSD37_5_12s.txt'; t = 12; SSD = "SSD37_5"

# sortér data i kolonner
data = np.loadtxt(filename)
f = data[0:,0]      # første kolonne = eksperiment nummer
y_ = data[0:,1]     # anden kolonne = y-kordinaten
x_ = data[0:,2]     # tredje kolonne = x-koordinaten
C = data[:,3:]      # fjerde til sidste kolonne = aflæst CHRG (nC)
D = data[:,3:]*K    # CHRG --> Dose ... nC --> Gy

# nogle praktiske størrelser
x = int(x_.max() + 1)   # antal målepunkter pr linje på perspexpladen = 5
y = int(y_.max() + 1)   # antal linjer med 5 målepunkter på perspexpladen = 6
N = y*x                 # antal målingspunkter på perspex-pladen = 30 (kan også være flere)
n = int(f.max() + 1)    # antal forsøg gjort ( = 3 hvis færdig)
m = C.shape[1]          # antal målinger pr punkt

#### UDREGN FEJLEN I D (∆D/D)^2 = (∆C/C)**2 + (∆K/K)**2 I HVERT MÅLEPUNKT ####
DD = np.zeros_like(D)   # fejlmatricen ∆D har samme form som dosismatricen D
for i in range(N*n):
    for j in range(m):
        DD[i,j] = np.sqrt( (DC/C[i,j])**2 + (DNK/NK)**2 )*D[i,j]

#### UDREGN GENNEMSNITTET AF MÅLINGERNE I HVERT PUNKT OG OMREGN TIL DOSIS ####
D_Matrix = np.zeros( (n, y, x) )                # 3 6x5-matricer med hvert enkelt forsøgs dosisgnsn i hver sin matrice
S_Matrix = np.zeros( (n, y, x) )                # 3 6x5-matricer med hvert enkelt forsøgs SD i hver sin matrice
for i in range(n):                              # gå igennem alle forsøg
    d_matrix = np.zeros( (y, x) )                   # 6x5-matrice: dosisgnsn som rækker
    s_matrix = np.zeros( (y, x) )                   # 6x5-matrice: SD som rækker
    for j in range(y):                          # gå igennem alle rækker
        yd = np.zeros(x)                                # 5-array: dosisgnsn for hvert punkt på en y-linje
        ys = np.zeros(x)                                # 5-array: SD for hvert punkt på en y-linje
        for k in range(x):                      # gå igennem alle målepunkter
            yd[k] = np.mean(D[k + j*x + i*N])   # gnsn af hvert punktsæt
            ys[k] = np.std(D[k + j*x + i*N])    # SD af hvert punktsæt
        d_matrix[j] = yd                        # 6x5-matrice med gnsn
        s_matrix[j] = ys                        # 6x5-matrice med SD
    D_Matrix[i] = d_matrix                      # 3 6x5 matricer med gnsn til PLOTTING
    S_Matrix[i] = s_matrix                      # 3 6x5 matricer med SD til PLOTTING

#### MATRICER MED GENNEMSNITSDOSIS, SE, OG SEM I HVERT PUNKT
Mean_Of_Total = np.zeros(N)
Mean_Of_Means = np.zeros(N)
SE = np.zeros(N)
SEM = np.zeros(N)
for i in range(N):                                          # gå igennem alle målepunkter på perspexpladen: range(N) = 0,1,2,...29
    AllPoints = np.zeros( (n, m) )                          # for hvert målepunkt; lav en 3-array med hver alle enkeltmålinger D i samme punkt
    AllErrors = np.zeros( (n, m) )                          # for hvert målepunkt; lav en 3-array med hver alle enkeltfejl ∆D i samme punkt
    PointMeans = np.zeros(n)                                # for hvert målepunkt; lav en 3-array med hver alle gennemsnitsmålinger <D> i samme punkt
    PointErrors = np.zeros(n)                               # for hvert målepunkt; lav en 3-array med hver alle gennemsnitsfejl <∆D> i samme punkt
    for j in range(n):                                      # gå igennem alle målinger i hvert punkt: range(n) = 0,1,2
        AllPoints[j] = D[i + N*j]                           # alle enkeltmålinger fra alle forsøg
        AllErrors[j] = DD[i + N*j]                          # alle enkeltmålingers standardfejl fra alle forsøg
        PointMeans[j] = np.mean(D[i + N*j])                 # gnsn af alle punktmålinger fra hvert forsøg
        PointErrors[j] = np.mean(DD[i + N*j])               # gnsn af alle punkters fejl fra hvert forsøg
    Mean_Of_Total[i] = np.mean(AllPoints)                   # N-matrice med gnsn af alle målinger i hvert målepunkt
    Mean_Of_Means[i] = np.mean(PointMeans)                  # N-matrice med gnsn af gnsn i hvert målepunkt (skal give samme som ovenstående)
    SE[i] = (np.std(AllPoints)/np.sqrt(n*m))                # SE i hvert målepunkt
    SEM[i] = (np.std(PointMeans)/np.sqrt(n))                # SEM i hvert målepunkt
    # SE[i] = np.sqrt( (np.std(AllPoints)/np.sqrt(n*m))**2 + np.mean(AllErrors)**2 )      # SE i hvert målepunkt
    # SEM[i] = np.sqrt( (np.std(PointMeans)/np.sqrt(n))**2 + np.mean(PointErrors)**2 )    # SEM i hvert målepunkt

ya = np.zeros(x)
ym = np.zeros(x)
yse = np.zeros(x)
ysem = np.zeros(x)
DA_Matrix = np.zeros( (y, x) )          # matrice til totalgennemsnit udregnet fra samtlige n*m målinger i hvert punkt
DM_Matrix = np.zeros( (y, x) )          # matrice til totalgennemsnit udregnet fra n gennemsnit i hvert punkt
SE_Matrix = np.zeros( (y, x) )          # matrice til SE i hvert punkt
SEM_Matrix = np.zeros( (y, x) )         # matrice til SEM i hvert punkt
for i in range(y):                      # gå igennem hver linje på perspexpladen
    for j in range(x):                  # gå igennem hvert punkt på perspexpladen
        ya[j] = Mean_Of_Total[j + i*x]  # fyld 6 5-arrays (én for hver linje på pladen) med gnsn af alle målinger
        ym[j] = Mean_Of_Means[j + i*x]  # fyld 6 5-arrays (én for hver linje på pladen) med gnsn af gennemsnittene fra hver forsøg
        yse[j]  = SE[j + i*x]           # fyld 6 5-arrays (én for hver linje på pladen) med standard error (SE)
        ysem[j] = SEM[j + i*x]          # fyld 6 5-arrays (én for hver linje på pladen) med standard error of the mean (SEM)
    DA_Matrix[i] = ya                   # fyld 6x5-dosematrix til plotting
    DM_Matrix[i] = ym                   # fyld 6x5-dosematrix til plotting
    SE_Matrix[i] = yse                  # fyld 6x5-SE-matrix til plotting
    SEM_Matrix[i] = ysem                # fyld 6x5-SEM-matrix til plotting

#### Sikkerhed for at SE og SEM er rigtig udregnet ####
if DA_Matrix.any() != DM_Matrix.any():
    print('ADVARSEL!! Gennemssnitsdose udregnet fra alle målinger ≠ gennemsnitsdose udregnet fra gennemsnittene')

#### NORMALISERING AF ALLE MÅLINGSINDEKS
#### FOR AT SE HVOR MEGET MERE RØRET GIVER
#### I DEN FØRSTE MÅLING
norm = np.zeros_like(D)             # norm
norm_wi = np.zeros_like(D)          # norm without index i
NormArray = np.zeros( (m, n*N) )    # NormArray
NormArray_wi = np.zeros( (m, n*N) ) # NormArray without index i

# udregn normaliserede værdier og fyld matricer til PLOTTING
for i in range(n*N):                                                        # hele datasættet igennem (90 punkter)
    norm[i] = D[i]/np.mean(D[i])                                            # normaliser alle målinger ift eget punktsæt
    for j in range(m):                                                      # 5 eller 10 gange, dvs en gang pr målingsindeks
        norm_wi[i] = D[i]/np.mean(np.concatenate( (D[i,:j], D[i,j+1:]) ))   # normaliser måling ift resterende målinger (ikke egen måling)
        NormArray[j,i] = norm[i,j]                                          # array med til plotting
        NormArray_wi[j,i] = norm_wi[i,j]                                    # array med til plotting

#### LILLE PLOT MED KALIBRERINGSFAKTOR FOR FORSKELLIGE EKSPONERINGSTIDER ####
first_meas_norms = np.array([1.022, 1.020, 1.012, 1.009 ])
first_meas_sems = np.array([0.006, 0.003, 0.003, 0.002 ])
scope = np.array([12, 13, 14, 20])

#### PLOTTING ####
#### PLOTTING ####
#### PLOTTING ####

#### STRÅLINGSINTENSITETPLOT ####
# intesitetsregulering i plottene så de bliver ensartede og sammenlignbare
# faste intensitetsværdier på alle fejlplot så de kan sammenlignes på tværs af forsøg
D_min = D_Matrix.min()
D_max = D_Matrix.max()
S_min = 1 #S_Matrix.min()
S_max = 3.3 #S_Matrix.max()

DM_min = D_min
DM_max = D_max
SE_min = S_min  # SE_Matrix.min()
SE_max = S_max  # SE_Matrix.max()
SEM_min = S_min # SEM_Matrix.min()
SEM_max = S_max # SEM_Matrix.max()

FS = 14 # fontsixe til superoverskrift
fs = 9  # fontsize til lengend()
titlecorrection = [0, 0.03, 1, 0.95]

titlea = "%s, %ss: %s målinger pr punkt"
if n == 1:
    figa, axa = plt.subplots(1,2,figsize=(7,4.5))
    figa.suptitle(titlea%(SSD, t, m), fontsize=FS)
    axa[0].imshow(D_Matrix[0], vmin=D_min, vmax=D_max, cmap='inferno',interpolation='lanczos')#, interpolation='lanczos')
    axa[0].set_title("Dosis: $\\overline{D}$ (mGy)")
    axa[0].invert_yaxis()
    axa[1].imshow(S_Matrix[0], vmin=S_min, vmax=S_max, cmap=plt.cm.Blues,interpolation='lanczos')#, interpolation='lanczos')
    axa[1].set_title("Standardafvig (mGy)")
    axa[1].invert_yaxis()
    for i in range(x):
        for j in range(y):
            c = D_Matrix[0,j,i]
            d = S_Matrix[0,j,i]
            axa[0].text(i, j, '%.1f' %c, va='bottom', ha='center', fontsize=fs)
            axa[0].text(i, j, "±"'%.1f' %d, va='top', ha='center', fontsize=fs)
            axa[1].text(i, j, '%.1f' %d, va='center', ha='center', fontsize=fs)
    plt.tight_layout(rect=[0, 0.03, 1, 0.9])
else:
    figa, axa = plt.subplots(nrows=2,ncols=n,figsize=(12, 8))
    figa.suptitle(titlea%(SSD, t, m), fontsize=FS)
    for i in range(n):
        axa[0,i].imshow(D_Matrix[i], vmin=D_min, vmax=D_max, cmap='inferno',interpolation='lanczos')
        axa[0,i].set_title("Dosis: $\\overline{D}$ (mGy), %s. forsøg"%(int(i+1)))
        axa[0,i].invert_yaxis()
        axa[1,i].imshow(S_Matrix[i], vmin=S_min, vmax=S_max, cmap=plt.cm.Blues,interpolation='lanczos')
        axa[1,i].set_title("Standardafvig (mGy)")
        axa[1,i].invert_yaxis()
        for j in range(x):
            for k in range(y):
                c = D_Matrix[i,k,j]
                d = S_Matrix[i,k,j]
                axa[0,i].text(j, k, '%.1f' %c, va='bottom', ha='center', fontsize=fs)
                axa[0,i].text(j, k, "±"'%.1f' %d, va='top', ha='center', fontsize=fs)
                axa[1,i].text(j, k, '%.1f' %d, va='center', ha='center', fontsize=fs)
    plt.tight_layout(rect=titlecorrection)
plt.savefig("%s_%s_dosiogSD.pdf"%(SSD,t))

#plot gennemsnitsdose og SEM fra alle forsøg
titleb = "%s, %ss: %s forsøg, %s målinger pr punkt "
figb, axb = plt.subplots(nrows=1,ncols=3,figsize=(9.55, 5))
figb.suptitle(titleb%(SSD, t,n, m), fontsize=FS)
axb[0].imshow(DM_Matrix, vmin=DM_min, vmax=DM_max, cmap='inferno',interpolation='lanczos')
axb[0].set_title("Dosis: $\\langle D\\rangle$ (mGy)")
axb[0].invert_yaxis()
axb[1].imshow(SE_Matrix, vmin=SE_min, vmax=SE_max, cmap=plt.cm.Blues,interpolation='lanczos')
axb[1].set_title("Standard error (mGy),\n SE = ${ \\frac{ SD(D) }{ \\sqrt{n\\cdot m} }} $  ")
axb[1].invert_yaxis()
axb[2].imshow(SEM_Matrix, vmin=SEM_min, vmax=SEM_max, cmap=plt.cm.Greens,interpolation='lanczos')
axb[2].set_title("Standard error of the mean (mGy),\n SEM = ${ \\frac{ SD(\overline{D}) }{ \\sqrt{n} } } $  ")
axb[2].invert_yaxis()
for i in range(x):
    for j in range(y):
        c = DM_Matrix[j,i]
        d = SE_Matrix[j,i]
        e = SEM_Matrix[j,i]
        g = np.array([c, d, e])
        for k in range(3):
            axb[k].text(i, j, '%.1f' %g[k], va='center', ha='center', fontsize=fs)
plt.tight_layout(rect=titlecorrection)
plt.savefig("%s_%s_gnsnSEogSEM.pdf"%(SSD,t))

#### PLOT FOR AT SE HVOR MEGET ENKELTE MÅLINGER AFVIGER FRA GENNEMSNITTET AF DE RESTERENDE MÅLINGER
# plt.rc('grid', color='w', linestyle='solid')
titlec = " %s, %ss: Alle i'ende-målinger i samme plot normaliseret i forhold til gennemsnittet af punktsættet"
ylimits = (0.91, 1.09)
if m <= 5:
    figc, axc = plt.subplots(nrows=1,ncols=5,figsize=(14, 6), sharey="all")
    figc.suptitle(titlec%(SSD, t), fontsize=FS)
    for i in range(m):
        avg = np.mean(NormArray[i])
        sem = np.std(NormArray[i])/np.sqrt(len(NormArray[i]))
        axc[i].axhline(y=avg,label="gnsn: %.3f±%.3f"%(avg,sem), color="red")
        axc[i].axhline(y=1, linestyle='dotted', color="lightgray")
        axc[i].plot(NormArray[i], ".", label="norm. måling")
        axc[i].set_title("alle %s.-målinger"%(i+1))
        axc[i].legend(loc=9, fontsize=fs)
        axc[i].set_ylim((ylimits))
        axc[i].set_xticks(np.linspace(0,int(n*N),10))
else:
    figc, axc = plt.subplots(nrows=2,ncols=5,figsize=(14, 8), sharey="all")
    figc.suptitle(titlec%(SSD, t), fontsize=FS)
    for i in range(2):
        for j in range(5):
            avg = np.mean(NormArray[j + i*5])
            sem = np.std(NormArray[j + i*5])/np.sqrt(len(NormArray[i]))
            axc[i,j].axhline(y=avg,label="gnsn: %.3f±%.3f"%(avg,sem), color="red")
            axc[i,j].axhline(y=1, linestyle='dotted', color="lightgray")
            axc[i,j].plot(NormArray[j + i*5], ".", label="norm. måling")
            axc[i,j].set_title("alle %s.-målinger"%(j + 1 + i*5))
            axc[i,j].legend(loc=9, fontsize=fs)
            axc[i,j].set_ylim(ylimits)
            axc[i,j].set_xlabel("Målingsrække")
            axc[i,j].set_ylabel("Normaliseret måling")
            axc[i,j].set_xticks(np.linspace(0,int(n*N),7))
plt.tight_layout(rect=titlecorrection)
plt.savefig("%s_%s_norm_rest_alle.pdf"%(SSD,t))


titled = "%s, %ss: Første måling i hvert punktsæt\n normaliseret i forhold til\n gennemsnittet af punktsættet"
figd, axd = plt.subplots()
axd.plot(NormArray[0], ".", label="normaliseret måling")
axd.set_ylim(ylimits)
axd.set_xlabel("Målingsindeks")
axd.set_ylabel("Normaliseret måling")
avg0 = np.mean(NormArray[0])
sem0 = np.std(NormArray[0])/np.sqrt(len(NormArray[0]))
# axd.fill_between(np.array(len(NormArray_wi[0])),avg_wi0-sem_wi0, avg_wi0+sem_wi0)
axd.axhline(y=avg0,label="gennemsnit: %.3f±%.3f"%(avg0, sem0), color="red")
axd.axhline(y=1, linestyle='dotted', color="lightgray")
axd.set_title(titled%(SSD, t), fontsize=FS)
axd.legend()
plt.tight_layout(rect=titlecorrection)
if n == 1:
    plt.xticks(np.linspace(0,int(n*N),7))
else:
    plt.xticks(np.linspace(0,int(n*N),19))

plt.tight_layout()
plt.savefig("%s_%s_norm_rest.pdf"%(SSD,t))

titlee = "Kalibreringsfaktor $\\delta(t)$ for kort eksponering"
fige, axe = plt.subplots()
axe.plot(scope,first_meas_norms, "r", label="$\\delta(t)$")
axe.fill_between(scope, first_meas_norms-first_meas_sems, first_meas_norms+first_meas_sems,alpha=0.3,label="SEM")
axe.set_xlabel("Eksponeringstid (s)")
axe.set_ylabel("Kalibreringsfaktor $\\delta(t)$")
axe.set_title(titlee, fontsize=FS)
axe.legend()
plt.tight_layout(rect=titlecorrection)
plt.savefig("Kalib_faktor_over_tid.pdf")

#### PRINT CHRG OG DOSE MÅLT I HVERT PUNKT ####
# for i in range(n*N):
#     if x[i]==0: #lav mellemrum i udskriften mellem hver kolonne på perspex-pladen
#         print()
#     print ("%s. forsøg: I feltet (%s,%s) er CHRG ="%( ( int(f[i] + 1), int(x[i]), int(y[i])  ) ),  "%.1f"%avg[i], '±', "%.1f"%std[i], "nC, og Dosen =", "%.1f"%dosav[i], "±", "%.1f"%dostd[i] , "mGy")
#### PRINT GENNEMSNITSDOSE OG SEM I HVERT MÅLEPUNKT ####
# for i in range(N):
#     if x[i]==0: #lav mellemrum i udskriften mellem hver kolonne på perspex-pladen
#         print()
#     print ("I feltet (%s,%s) er dosen i gennemsnit ="  %( int(x[i]), int(y[i]) ),"%.1f"%DOSE_AVG[i], "±","%.1f"%DOSE_SEM[i], "mGy")
# print()
print()
print("D[0] =", D[0])
print("n*N =", n*N)
print("m =", m)
print ("x =", x )
print ("y =", y)
print ("N = y*x =", N)
print ("n =", n)
print( "n*x*y =", n*x*y)

print()
print("antal forsøg kørt: %s " %(n))
print()

plt.show()
