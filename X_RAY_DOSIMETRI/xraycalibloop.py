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




#### KONSTANTER ####
#omregningsfaktor K for dosimeteret i mGy/nC (milliGray pr nanoCoulomb)
#faktorer findes i IAEA-rapporten TRS 227 (Andreo et al. 1987)
ku = 1                  #corrects for the change in response as the spectral distribution changes when moving from air to water
muen_over_rho = 1.075   #ratio of the mass energy absorption coefficient between water and air, averaged over the photon spectrum at reference depth
pu = 1.02               #perturbation factor
kTP = 1.018             #corrects for the change in temperature and pressure from the calibration to the measurement
NK = 43.77              #air kerma calibration factor for the given beam quality
K = NK*ku*muen_over_rho*pu*kTP

DeltaNK = 0.39
DeltaC = 0.02


print(100/K)

#### VÆLG TXT.-FIL SOM SKAL KØRES ####
# filename = 'xray_data_SSD50_20s.txt'; t = 20; SSD = "SSD50"
# filename = 'xray_data_SSD40_14s.txt'; t = 14; SSD = "SSD40"
filename = 'xray_data_SSD40_13s.txt'; t = 13; SSD = "SSD40"
# filename = 'xray_data_SSD37_5_12s.txt'; t = 12; SSD = "SSD37.5"

data = np.loadtxt(filename)

#sortér data i kolonner
f = data[0:,0]      #første kolonne = eksperiment nummer
y = data[0:,1]      #anden kolonne = y-kordinaten
x = data[0:,2]      #tredje kolonne = x-koordinaten
m = data[:,3:]*K    #kolonner med målinger, omregnet fra nC til Gy

x_points = int(x.max() + 1) #antal målepunkter pr linje = 5
y_lines = int(y.max() + 1)  #antal linjer med målepunkter = 6
N = y_lines*x_points        #antal målingspunkter på perspex-pladen = 30 (kan også være flere)
n = int(f.max() + 1)        #antal forsøg gjort ( = 3 hvis færdig)



print("m[0] =", m[0])
print("m.shape[0] =", m.shape[0])
print("m.shape[1] =", m.shape[1])
# print ("n*m.shape[1] =", n*m.shape[1])
print ("x_points =", x_points )
print ("y_lines =", y_lines)
print ("N = y_lines*x_points =", N)
print ("n =", n)

#### REGN GENNEMSNITTET AF MÅLINGERNE I HVERT PUNKT OG OMREGN TIL DOSIS ####
ya = np.zeros(x_points)                     #dosisgennemsnit for hver y-linje
yo = np.zeros(x_points)                     #standardafvig for hver y-linje
a_matrix = np.zeros( (y_lines, x_points) )    #5x6-matrice med dosisgennemsnittene som rækker
o_matrix = np.zeros( (y_lines, x_points) )    #5x6-matrice med standardafvigene som rækker
A_matrix = np.zeros( (n, y_lines, x_points) ) #3x6x5-matrice med alle forsøgs dosisgennemsnit
O_matrix = np.zeros( (n, y_lines, x_points) ) #3x6x5-matrice med alle forsøgs standardafvig
for i in range(n): #3                               #gå igennem alle forsøg
    for j in range(y_lines): #6                     #gå igennem alle rækker
        for k in range(x_points): #5                #gå igennem alle målepunkter
            ya[k] = np.mean(m[k + j*x_points + i*N])
            yo[k] = np.std(m[k + j*x_points + i*N])
        a_matrix[j] = ya
        o_matrix[j] = yo
    A_matrix[i] = a_matrix
    O_matrix[i] = o_matrix

#### MATRIX MED GENNEMSNITSDOSIS OG STANDARD ERROR
Mean_Of_Total = np.zeros(N)
Mean_Of_Means = np.zeros(N)
SE = np.zeros(N)
SEM = np.zeros(N)

AllPoints = np.zeros( (n, m.shape[1]) ) #samtlige enkeltmålinger i hvert punkt

for i in range(N):                                  # gå igennem alle målepunkter: range(N) = 0,1,2,...29
    PointMeans = np.zeros(n)                        # for hvert målepunkt; lav en 3-array med hver alle målinger i samme punkt
    for j in range(n):                              # gå igennem alle målinger i hvert punkt: range(n) = 0,1,2
        AllPoints[j] = m[i + N*j]
        PointMeans[j] = np.mean(m[i + N*j])         # fyld n-array med alle gennemsnit i samme punkt
    Mean_Of_Total[i] = np.mean(AllPoints)           # lav N-array med gennemsnit fra samtlige målinger
    Mean_Of_Means[i] = np.mean(PointMeans)          # lav 30-array med gennemsnit af gennemsnitsdoser i hvert målepunkt
    SE[i] = np.std(AllPoints)/np.sqrt(n*m.shape[1]) # lav 30-array med SEM i hvert målepunkt SEM=std(all_meas)/sqrt(30)
    SEM[i] = np.std(PointMeans)/np.sqrt(n)          # lav 30-array med SEM i hvert målepunkt SEM=std(all_avgs)/sqrt(3)

y_all = np.zeros(x_points)
y_means = np.zeros(x_points)
y_se = np.zeros(x_points)
y_sem = np.zeros(x_points)
DoseMatrix_all = np.zeros((y_lines, x_points))
DoseMatrix_mean = np.zeros((y_lines, x_points))
SE_Matrix = np.zeros((y_lines, x_points))
SEM_Matrix = np.zeros((y_lines, x_points))

for i in range(y_lines):
    for j in range(x_points):
        y_all[j]  = Mean_Of_Total[j + i*(x_points)]
        y_se[j]  = SE[j + i*(x_points)]
        y_means[j] = Mean_Of_Means[j + i*(x_points)]
        y_sem[j] = SEM[j + i*(x_points)]
    DoseMatrix_all[i] = y_all
    DoseMatrix_mean[i] = y_means
    SE_Matrix[i] = y_se
    SEM_Matrix[i] = y_sem

#### Sikkerhed for at SE og SEM er rigtig udregnet ####
if DoseMatrix_all.any() != DoseMatrix_mean.any():
    print('ADVARSEL!! Gennemssnitsdose udregent fra alle målinger ≠ gennemsnitsdose udregnet fra gennemsnittene')



#### NORMALISERING AF ALLE MÅLINGSINDEKS
#### FOR AT SE HVOR MEGET MERE RØRET GIVER
#### I DEN FØRSTE MÅLING
norm = np.zeros(m.shape)
norm_wi = np.zeros(m.shape)
NormArray = np.zeros( (m.shape[1], m.shape[0]) )
NormArray_wi = np.zeros( (m.shape[1], m.shape[0]) )

for i in range(m.shape[0]):                                                 # hele datasættet igennem (90 punkter)
    norm[i] = m[i]/(np.mean(m[i]))                                          # normaliser alle målinger ift eget punktsæt
    for j in range(m.shape[1]):                                             # 5 eller 10 gange, dvs en gang pr målingsindeks
        norm_wi[i] = m[i]/np.mean(np.concatenate((m[i][:j], m[i][j+1:])))   # normaliser måling ift resterende målinger (ikke egen måling)
        NormArray[j,i] = norm[i,j]                                          # array til plotting
        NormArray_wi[j,i] = norm_wi[i,j]                                    # array til plotting

#### PLOTTING ####
# plt.rcParams.update({'font.size': 9})

#### STRÅLINGSINTENSITETPLOT ####
#intesitetsregulering i plottene så de bliver ensartede og sammenlignbare
A_min = A_matrix.min()
A_max = A_matrix.max()
O_min = 0#O_matrix.min()
O_max = 3.5#O_matrix.max()

nx1 = m.shape[1]    #antal målinger pr punkt i et forsøg
nxn = n*m.shape[1]  #antal målinger pr punkt i alle forsøg

f = 9
#plot enkelte forsøg i samme figur
if n == 1:
    figs, axs = plt.subplots(1,2)
    figs.suptitle("Målt dosis med standardafvig, %s målinger pr punkt, %s, %ss"%(nx1, SSD, t), fontsize=14)
    axs[0].imshow(A_matrix[0], vmin=A_min, vmax=A_max, cmap='inferno',interpolation='lanczos')#, interpolation='lanczos')
    axs[0].set_title("Målt dosis (mGy)")
    axs[0].invert_yaxis()
    axs[1].imshow(O_matrix[0], vmin=O_min, vmax=O_max, cmap=plt.cm.Blues,interpolation='lanczos')#, interpolation='lanczos')
    axs[1].set_title("Standardafvig (mGy)")
    axs[1].invert_yaxis()
    for i in range(x_points):
        for j in range(y_lines):
            c = A_matrix[0,j,i]
            d = O_matrix[0,j,i]
            axs[0].text(i, j, '%.2f' %c, va='bottom', ha='center', fontsize=f)
            axs[0].text(i, j, "±"'%.2f' %d, va='top', ha='center', fontsize=f)
            axs[1].text(i, j, '%.2f' %d, va='center', ha='center', fontsize=f)
else:
    figs, axs = plt.subplots(nrows=2,ncols=n,figsize=(12, 8))
    figs.suptitle("Målt dosis med standardafvig fra hvert forsøg, %s målinger pr punkt, %s, %ss"%(nx1, SSD, t), fontsize=14)
    for i in range(n):
        axs[0,i].imshow(A_matrix[i], vmin=A_min, vmax=A_max, cmap='inferno',interpolation='lanczos')
        axs[0,i].set_title("Målt dosis (mGy), %s. forsøg"%(int(i+1)))
        axs[0,i].invert_yaxis()
        axs[1,i].imshow(O_matrix[i], vmin=O_min, vmax=O_max, cmap=plt.cm.Blues,interpolation='lanczos')
        axs[1,i].set_title("Standardafvig (mGy)")
        axs[1,i].invert_yaxis()
        for j in range(x_points):
            for k in range(y_lines):
                c = A_matrix[i,k,j]
                d = O_matrix[i,k,j]
                axs[0,i].text(j, k, '%.2f' %c, va='bottom', ha='center', fontsize=f)
                axs[0,i].text(j, k, "±"'%.2f' %d, va='top', ha='center', fontsize=f)
                axs[1,i].text(j, k, '%.2f' %d, va='center', ha='center', fontsize=f)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# plt.show()

#plot gennemsnitsdose og SEM fra alle forsøg
fig, ax = plt.subplots(nrows=1,ncols=3,figsize=(9.55, 5))
fig.suptitle("Gennemsnitsdosis og standard error, %s $\\times$ %s målinger pr punkt, %s, %ss"%(n, nx1, SSD, t), fontsize=14)
ax[0].imshow(DoseMatrix_mean, vmin=A_min, vmax=A_max, cmap='inferno',interpolation='lanczos')
ax[0].set_title("Gennemsnitsdosis (mGy)")
ax[0].invert_yaxis()
ax[1].imshow(SE_Matrix, vmin=O_min, vmax=O_max, cmap=plt.cm.Blues,interpolation='lanczos')
ax[1].set_title("Standard error (mGy),\n$SE=\\frac{np.std([x_1,...,x_{%s}])}{√%s}$  "%(nxn, nxn))
ax[1].invert_yaxis()
ax[2].imshow(SEM_Matrix, vmin=O_min, vmax=O_max, cmap=plt.cm.Greens,interpolation='lanczos')
ax[2].set_title("Standard error of the mean (mGy),\n $SEM=\\frac{np.std([\\overline{X}_1,...,\\overline{X}_{%s}])}{√%s}$ " %(n,n))
ax[2].invert_yaxis()
for i in range(x_points):
    for j in range(y_lines):
        c = DoseMatrix_mean[j,i]
        d = SE_Matrix[j,i]
        e = SEM_Matrix[j,i]
        g = np.array([c, d, e])
        for k in range(3):
            ax[k].text(i, j, '%.2f' %g[k], va='center', ha='center', fontsize=f)
plt.tight_layout(rect=[0, 0.03, 1, 0.93])







#### PLOT FOR AT SE HVOR MEGET ENKELTE MÅLINGER AFVIGER FRA GENNEMSNITTET

tito = "Målingsindeks normaliseret i forhold til gennemsnittet i hvert punktmålingssæt, %s, %ss"
ylimits = (0.91, 1.09)
size = n*N
if m.shape[1] <= 5:
    figo, axo = plt.subplots(nrows=1,ncols=5,figsize=(14, 6), sharey="all")
    figo.suptitle(tito%(SSD, t), fontsize=14)
    for i in range(m.shape[1]):
        k = np.mean(NormArray[i])
        axo[i].axhline(y=k,label="gennemsnit: %.4f"%k, color="red")
        axo[i].plot(NormArray[i], ".", label="målinger")
        axo[i].set_title("alle %s.-målinger"%(i+1))
        axo[i].legend(loc=9, fontsize=f)
        axo[i].set_ylim((ylimits))
        axo[i].set_xticks(np.linspace(0,int(n*N),10))
else:
    figo, axo = plt.subplots(nrows=2,ncols=5,figsize=(14, 8), sharey="all")
    figo.suptitle(tito%(SSD, t), fontsize=14)
    for i in range(2):
        for j in range(5):
            k = np.mean(NormArray[j + i*5])
            axo[i,j].axhline(y=k,label="gennemsnit: %.4f"%k, color="red")
            axo[i,j].plot(NormArray[j + i*5], ".", label="målinger")
            axo[i,j].set_title("alle %s.-målinger"%(j + 1 + i*5))
            axo[i,j].legend(loc=9, fontsize=f)
            axo[i,j].set_ylim(ylimits)
            axo[i,j].set_xlabel("Målingsrække")
            axo[i,j].set_ylabel("Normaliseret måling")
            axo[i,j].set_xticks(np.linspace(0,int(n*N),7))
plt.tight_layout(rect=[0, 0.03, 1, 0.95])




titi = "Første måling normailseret i forhold til\ngennemsnittet i hvert punktmålingssæt, %s, %ss"
figi, axi = plt.subplots()
axi.plot(NormArray[0], ".", label="normaliseret måling")
axi.set_ylim(ylimits)
axi.set_xlabel("Målingsindeks")
axi.set_ylabel("Normaliseret måling")
k = np.mean(NormArray[0])
axi.axhline(y=k,label="gennemsnitlig førstemåling ift. gennemsnittet: %.4f"%k, color="red")
axi.set_title(titi%(SSD, t))
axi.legend()
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
if n == 1:
    plt.xticks(np.linspace(0,int(n*N),7))
else:
    plt.xticks(np.linspace(0,int(n*N),19))



titw = "Første måling normailseret i forhold til\ngennemsnittet af de resterende målinger\ni hvert punktmålingssæt, %s, %ss"
figw, axw = plt.subplots()
axw.plot(NormArray_wi[0], ".", label="normaliseret måling")
axw.set_ylim(ylimits)
axw.set_xlabel("Målingsindeks")
axw.set_ylabel("Normaliseret måling")
k_wi = np.mean(NormArray_wi[0])
axw.axhline(y=k_wi,label="gennemsnitlig førstemåling ift. gennemsnittet: %.4f"%k_wi, color="red")
axw.set_title(titw%(SSD, t))
axw.legend()
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
if n == 1:
    plt.xticks(np.linspace(0,int(n*N),7))
else:
    plt.xticks(np.linspace(0,int(n*N),19))


plt.show()



#### PRINT CHRG OG DOSE MÅLT I HVERT PUNKT ####
# for i in range(m.shape[0]):
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
print("antal forsøg kørt: %s " %(n))
print()
