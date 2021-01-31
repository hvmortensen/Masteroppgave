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
# filename = 'xray_data_SSD50_20s.txt'; t = 20; SSD = "SSD50"
# filename = 'xray_data_SSD40_14s.txt'; t = 14; SSD = "SSD40"
filename = 'xray_data_SSD40_13s.txt'; t = 13; SSD = "SSD40"
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
# def DD(c_,d_):
#     return np.sqrt( (DC/c_)**2 + (DNK/NK)**2 )*d_
DD = np.zeros_like(C)
for i in range(N*n):
    for j in range(m):
        DD[i,j] = np.sqrt( (DC/C[i,j])**2 + (DNK/NK)**2 )*D[i,j]


#### UDREGN GENNEMSNITTET AF MÅLINGERNE I HVERT PUNKT OG OMREGN TIL DOSIS ####
yd = np.zeros(x)                                # 5-array: dosisgennemsnit for hver y-linje
ys = np.zeros(x)                                # 5-array: standardafvig for hver y-linje
d_matrix = np.zeros( (y, x) )                   # 6x5-matrice: dosisgennemsnittene som rækker
s_matrix = np.zeros( (y, x) )                   # 6x5-matrice: standardafvigene som rækker
D_Matrix = np.zeros( (n, y, x) )                # 3 6x5-matricer med hvert enkelt forsøgs dosisgennemsnit i hver sin matrice
S_Matrix = np.zeros( (n, y, x) )                # 3 6x5-matricer med hvert enkelt forsøgs standardafvig i hver sin matrice

# udregn gnsn og SD og fyld matricer til PLOTTING
for i in range(n):                              # gå igennem alle forsøg
    for j in range(y):                          # gå igennem alle rækker
        for k in range(x):                      # gå igennem alle målepunkter
            yd[k] = np.mean(D[k + j*x + i*N])   # gennemsnit af hvert punktsæt
            ys[k] = np.std(D[k + j*x + i*N])    # standardafvig af hvert punktsæt
        d_matrix[j] = yd                        # 6x5-matrice med gennemsnit
        s_matrix[j] = ys                        # 6x5-matrice med standardafvig
    D_Matrix[i] = d_matrix                      # 3 6x5 matricer med gennemsnit til PLOTTING
    S_Matrix[i] = s_matrix                      # 3 6x5 matricer med standardafvig til PLOTTING

#### MATRICER MED GENNEMSNITSDOSIS, SE, OG SEM
Mean_Of_Total = np.zeros(N)
Mean_Of_Means = np.zeros(N)
SE = np.zeros(N)
SEM = np.zeros(N)
AllPoints = np.zeros( (n, m) )              #samtlige enkeltmålinger i hvert punkt
AllErrors = np.zeros( (n, m) )              #samtlige enkeltmålingers standardfejl i hvert punkt

# udregn gnsn, SE, og SEM
for i in range(N):                                          # gå igennem alle målepunkter på perspexpladen: range(N) = 0,1,2,...29
    PointMeans = np.zeros(n)                                # for hvert målepunkt; lav en 3-array med hver alle målinger D i samme punkt
    PointErrors = np.zeros(n)                               # for hvert målepunkt; lav en 3-array med hver alle standardfejl ∆D i samme punkt
    for j in range(n):                                      # gå igennem alle målinger i hvert punkt: range(n) = 0,1,2
        AllPoints[j] = D[i + N*j]                           # alle enkeltmålinger fra alle forsøg
        AllErrors[j] = DD[i + N*j]                          # alle enkeltmålingers standardfejl fra alle forsøg
        PointMeans[j] = np.mean(D[i + N*j])                 # fyld n-array med alle gennemsnitsmålinger i samme punkt for hver n
        PointErrors[j] = np.mean(DD[i + N*j])               # fyld n-array med alle gennemsnits-standardfejl i samme punkt for hver n
    Mean_Of_Total[i] = np.mean(AllPoints)                   # udregn gennemsnit lav N-array med gennemsnit fra samtlige målinger
    Mean_Of_Means[i] = np.mean(PointMeans)                  # udregn gennemsnit lav N-array med gennemsnit af gennemsnitsdoser i hvert målepunkt
    SE[i] = np.sqrt( (np.std(AllPoints)/np.sqrt(n*m))**2\
                    + np.mean(AllErrors)**2 )               # udregn SE og lav N-array med SEM i hvert målepunkt
    SEM[i] = np.sqrt( (np.std(PointMeans)/np.sqrt(n))**2\
                    + np.mean(PointErrors)**2 )             # udregn SEM og lav N-array med SEM i hvert målepunkt

ya = np.zeros(x)
ym = np.zeros(x)
yse = np.zeros(x)
ysem = np.zeros(x)
DA_Matrix = np.zeros( (y, x) )
DM_Matrix = np.zeros( (y, x) )
SE_Matrix = np.zeros( (y, x) )
SEM_Matrix = np.zeros( (y, x) )

# fyld matricer til PLOTTING
for i in range(y):                          # gå igennem hver linje på perspexpladen
    for j in range(x):                      # gå igennem hvert punkt på perspexpladen
        ya[j] = Mean_Of_Total[j + i*(x)]    # fyld 6 arrays med gennemsnittene af alle målinger
        ym[j] = Mean_Of_Means[j + i*(x)]    # fyld 6 arrays med gennemsnit af gennemsnittene fra hver forsøg
        yse[j]  = SE[j + i*(x)]             # fyld 6 arrays med standard error (SE)
        ysem[j] = SEM[j + i*(x)]            # fyld 6 arrays med standard error of the mean (SEM)
    DA_Matrix[i] = ya                       # fyld 6x5-dosematrix til plotting
    DM_Matrix[i] = ym                       # fyld 6x5-dosematrix til plotting
    SE_Matrix[i] = yse                      # fyld 6x5-SE-matrix til plotting
    SEM_Matrix[i] = ysem                    # fyld 6x5-SEM-matrix til plotting

#### Sikkerhed for at SE og SEM er rigtig udregnet ####
if DA_Matrix.any() != DM_Matrix.any():
    print('ADVARSEL!! Gennemssnitsdose udregent fra alle målinger ≠ gennemsnitsdose udregnet fra gennemsnittene')

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
first_meas_norms = np.array([1.024, 1.023, 1.015, 1.011 ])
first_meas_sems = np.array([0.007, 0.003, 0.004, 0.002 ])
scope = np.array([12, 13, 14, 20])

#### PLOTTING ####
#### PLOTTING ####
#### PLOTTING ####

#### STRÅLINGSINTENSITETPLOT ####
#intesitetsregulering i plottene så de bliver ensartede og sammenlignbare
D_min = D_Matrix.min()
D_max = D_Matrix.max()
S_min = 0#S_Matrix.min()
S_max = 3.5#S_Matrix.max()

FS = 14 # fontsixe til superoverskrift
fs = 9  # fontsize til lengend()
titlecorrection = [0, 0.03, 1, 0.95]

titlea = "%s, %ss: %s målinger pr punkt"
if n == 1:
    figa, axa = plt.subplots(1,2,figsize=(7,4.5))
    figa.suptitle(titlea%(SSD, t, m), fontsize=FS)
    axa[0].imshow(D_Matrix[0], vmin=D_min, vmax=D_max, cmap='inferno',interpolation='lanczos')#, interpolation='lanczos')
    axa[0].set_title("Målt dosis (mGy)")
    axa[0].invert_yaxis()
    axa[1].imshow(S_Matrix[0], vmin=S_min, vmax=S_max, cmap=plt.cm.Blues,interpolation='lanczos')#, interpolation='lanczos')
    axa[1].set_title("Standardafvig (mGy)")
    axa[1].invert_yaxis()
    for i in range(x):
        for j in range(y):
            c = D_Matrix[0,j,i]
            d = S_Matrix[0,j,i]
            axa[0].text(i, j, '%.2f' %c, va='bottom', ha='center', fontsize=fs)
            axa[0].text(i, j, "±"'%.2f' %d, va='top', ha='center', fontsize=fs)
            axa[1].text(i, j, '%.2f' %d, va='center', ha='center', fontsize=fs)
    plt.tight_layout(rect=[0, 0.03, 1, 0.9])
else:
    figa, axa = plt.subplots(nrows=2,ncols=n,figsize=(12, 8))
    figa.suptitle(titlea%(SSD, t, m), fontsize=FS)
    for i in range(n):
        axa[0,i].imshow(D_Matrix[i], vmin=D_min, vmax=D_max, cmap='inferno',interpolation='lanczos')
        axa[0,i].set_title("Målt dosis (mGy), %s. forsøg"%(int(i+1)))
        axa[0,i].invert_yaxis()
        axa[1,i].imshow(S_Matrix[i], vmin=S_min, vmax=S_max, cmap=plt.cm.Blues,interpolation='lanczos')
        axa[1,i].set_title("Standardafvig (mGy)")
        axa[1,i].invert_yaxis()
        for j in range(x):
            for k in range(y):
                c = D_Matrix[i,k,j]
                d = S_Matrix[i,k,j]
                axa[0,i].text(j, k, '%.2f' %c, va='bottom', ha='center', fontsize=fs)
                axa[0,i].text(j, k, "±"'%.2f' %d, va='top', ha='center', fontsize=fs)
                axa[1,i].text(j, k, '%.2f' %d, va='center', ha='center', fontsize=fs)
    plt.tight_layout(rect=titlecorrection)
plt.savefig("%s_%s_dosiogSD.pdf"%(SSD,t))

#plot gennemsnitsdose og SEM fra alle forsøg
titleb = "%s, %ss: %s forsøg, %s målinger pr punkt "
figb, axb = plt.subplots(nrows=1,ncols=3,figsize=(9.55, 5))
figb.suptitle(titleb%(SSD, t,n, m), fontsize=FS)
axb[0].imshow(DM_Matrix, vmin=D_min, vmax=D_max, cmap='inferno',interpolation='lanczos')
axb[0].set_title("Gennemsnitsdosis (mGy)")
axb[0].invert_yaxis()
axb[1].imshow(SE_Matrix, vmin=S_min, vmax=S_max, cmap=plt.cm.Blues,interpolation='lanczos')
axb[1].set_title("Standard error (mGy),\n SE = $\\sqrt{ (\\frac{ SD(D_i) }{ \\sqrt{n\\cdot m} })^2 + \\overline{∆D}^2} $  ")
#SE = $\\sqrt{\\frac{SD(x_i)^2 + (\overline{∆D})^2}{{m\\cdot n}}}$")
#SE = $\\frac{np.std([x_1,...,x_{%s}])}{√%s}$  "%(int(n*m), int(n*m)))
axb[1].invert_yaxis()
axb[2].imshow(SEM_Matrix, vmin=S_min, vmax=S_max, cmap=plt.cm.Greens,interpolation='lanczos')
axb[2].set_title("Standard error of the mean (mGy),\n SEm = $\\sqrt{ (\\frac{ SD(\overline{D}_j) }{ \\sqrt{n} })^2 + \\langle\overline{∆D}\\rangle^2} $  ")
#SEM = $\\sqrt{\\frac{SD(\overline{X}_j)^2 + (\overline{∆D})^2}{{n}}}$")
#SEM = $\\frac{np.std([\\overline{X}_1,...,\\overline{X}_{%s}])}{√%s}$ " %(n,n))
axb[2].invert_yaxis()
for i in range(x):
    for j in range(y):
        c = DM_Matrix[j,i]
        d = SE_Matrix[j,i]
        e = SEM_Matrix[j,i]
        g = np.array([c, d, e])
        for k in range(3):
            axb[k].text(i, j, '%.2f' %g[k], va='center', ha='center', fontsize=fs)
plt.tight_layout(rect=titlecorrection)
plt.savefig("%s_%s_gnsnSEogSEM.pdf"%(SSD,t))


#### PLOT FOR AT SE HVOR MEGET ENKELTE MÅLINGER AFVIGER FRA GENNEMSNITTET AF DE RESTERENDE MÅLINGER

# plt.rc('grid', color='w', linestyle='solid')
titlec = " %s, %ss: m'te måling i hvert punktsæt normaliseret i forhold til gennemsnittet af de resterende målinger i samme sæt"
ylimits = (0.91, 1.09)
if m <= 5:
    figc, axc = plt.subplots(nrows=1,ncols=5,figsize=(14, 6), sharey="all")
    figc.suptitle(titlec%(SSD, t), fontsize=FS)
    for i in range(m):
        avg_wi = np.mean(NormArray_wi[i])
        sem_wi = np.std(NormArray_wi[i])/np.sqrt(len(NormArray_wi[i]))
        axc[i].axhline(y=avg_wi,label="gnsn: %.3f±%.3f"%(avg_wi,sem_wi), color="red")
        axc[i].plot(NormArray_wi[i], ".", label="norm. måling")
        axc[i].set_title("alle %s.-målinger"%(i+1))
        axc[i].legend(loc=9, fontsize=fs)
        axc[i].set_ylim((ylimits))
        axc[i].set_xticks(np.linspace(0,int(n*N),10))
else:
    figc, axc = plt.subplots(nrows=2,ncols=5,figsize=(14, 8), sharey="all")
    figc.suptitle(titlec%(SSD, t), fontsize=FS)
    for i in range(2):
        for j in range(5):
            avg_wi = np.mean(NormArray_wi[j + i*5])
            sem_wi = np.std(NormArray_wi[j + i*5])/np.sqrt(len(NormArray_wi[i]))
            axc[i,j].axhline(y=avg_wi,label="gnsn: %.3f±%.3f"%(avg_wi,sem_wi), color="red")
            axc[i,j].plot(NormArray_wi[j + i*5], ".", label="norm. måling")
            axc[i,j].set_title("alle %s.-målinger"%(j + 1 + i*5))
            axc[i,j].legend(loc=9, fontsize=fs)
            axc[i,j].set_ylim(ylimits)
            axc[i,j].set_xlabel("Målingsrække")
            axc[i,j].set_ylabel("Normaliseret måling")
            axc[i,j].set_xticks(np.linspace(0,int(n*N),7))
plt.tight_layout(rect=titlecorrection)
plt.savefig("%s_%s_norm_rest_alle.pdf"%(SSD,t))


titled = "%s, %ss: Første måling i hvert punktsæt\n normaliseret i forhold til gennemsnittet\n af de resterende målinger i samme sæt"
figd, axd = plt.subplots()
axd.plot(NormArray_wi[0], ".", label="normaliseret måling")
axd.set_ylim(ylimits)
axd.set_xlabel("Målingsindeks")
axd.set_ylabel("Normaliseret måling")
avg_wi0 = np.mean(NormArray_wi[0])
sem_wi0 = np.std(NormArray_wi[0])/np.sqrt(len(NormArray_wi[0]))
# axd.fill_between(np.array(len(NormArray_wi[0])),avg_wi0-sem_wi0, avg_wi0+sem_wi0)
axd.axhline(y=avg_wi0,label="gennemsnit: %.3f±%.3f"%(avg_wi0, sem_wi0), color="red")
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
#     print ("%s. forsøg: I feltet (%s,%s) er CHRG ="%( ( int(f[i] + 1), int(x[i]), int(y[i])  ) ),  "%.2f"%avg[i], '±', "%.2f"%std[i], "nC, og Dosen =", "%.2f"%dosav[i], "±", "%.2f"%dostd[i] , "mGy")
#### PRINT GENNEMSNITSDOSE OG SEM I HVERT MÅLEPUNKT ####
# for i in range(N):
#     if x[i]==0: #lav mellemrum i udskriften mellem hver kolonne på perspex-pladen
#         print()
#     print ("I feltet (%s,%s) er dosen i gennemsnit ="  %( int(x[i]), int(y[i]) ),"%.2f"%DOSE_AVG[i], "±","%.2f"%DOSE_SEM[i], "mGy")
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
