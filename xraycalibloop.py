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


Lidt teori:
Hvornår skal jeg bruge standardafvig (SD = std()) og hvornår skal jeg bruge
standard error of the mean (SEM = SD/√n)?

Investopedia.com skriver:

"The standard deviation (SD) measures the amount of variability, or dispersion,
from the individual data values to the mean, while the standard error of the
mean (SEM) measures how far the sample mean (average) of the data is likely to
be from the true population mean. The SEM is always smaller than the SD.

KEY TAKEAWAYS
Standard deviation (SD) measures the dispersion of a dataset relative to its
mean.
Standard error of the mean (SEM) measured how much discrepancy there is likely
to be in a sample's mean compared to the population mean.
The SEM takes the SD and divides it by the square root of the sample size.

SEM vs. SD
Standard deviation and standard error are both used in all types of statistical
studies, including those in finance, medicine, biology, engineering, psychology,
etc. In these studies, the standard deviation (SD) and the estimated standard
error of the mean (SEM) are used to present the characteristics of sample data
and to explain statistical analysis results. However, some researchers
occasionally confuse the SD and SEM. Such researchers should remember that the
calculations for SD and SEM include different statistical inferences, each of
them with its own meaning. SD is the dispersion of individual data values.

In other words, SD indicates how accurately the mean represents sample data.
However, the meaning of SEM includes statistical inference based on the
sampling distribution. SEM is the SD of the theoretical distribution of the
sample means (the sampling distribution)."


Bemærkninger:
Jeg bør nok køre mindst ti målinger i hvert punkt siden nogle af målingerne ikke
holder sig indenfor standardafviget.
"""

import numpy as np
import matplotlib.pyplot as plt

#omregningsfaktor K for dosimeteret i mGy/nC (milliGray pr nanoCoulomb)
#faktorer findes i IAEA-rapporten TRS 227 (Andreo et al. 1987)
ku = 1                  #corrects for the change in response as the spectral distribution changes when moving from air to water
muen_over_rho = 1.075   #ratio of the mass energy absorption coefficient between water and air, averaged over the photon spectrum at reference depth
pu = 1.02               #perturbation factor
kTP = 1.018             #corrects for the change in temperature and pressure from the calibration to the measurement
NK = 43.77              #air kerma calibration factor for the given beam quality
K = NK*ku*muen_over_rho*pu*kTP

# filename = 'xray_data_SSD50_20s.txt'
# filename = 'xray_data_SSD40_14s.txt'
filename = 'xray_data_SSD40_13s.txt'
# filename = 'metadat.txt'


#tekstfilen som indeholder data fra samtlige eksperimenter
data = np.loadtxt(filename)

#sortér data i kolonner
f = data[0:,0]  #første kolonne = eksperiment nummer
y = data[0:,1]  #anden kolonne = y-kordinaten
x = data[0:,2]  #tredje kolonne = x-koordinaten
m = data[:,3:]  #kolonner med målinger

x_points = int(x.max() + 1) #antal målepunkter pr linje = 5
y_lines = int(y.max() + 1)  #antal linjer med målepunkter = 6
N = y_lines*x_points        #antal målingspunkter på perspex-pladen = 30 (kan også være flere)
n = int(f.max() + 1)        #antal forsøg gjort ( = 3 hvis færdig)

#### GENNEMSNIT MED STANDARDAFVIG & DOSIS ####

#dan tomme arrays til gennemsnit, standardafvig og udledet dosis
avg = np.zeros(n*N) #altid n*N = 30 (eller 56) så fyldt med nuller hvis forsøget ikke er afsluttet
std = np.zeros(n*N)
dosav = np.zeros(n*N)
dostd = np.zeros(n*N)
#fyld arrays'ne med data og print
for i in range(m.shape[0]): #for i in (0,1,2,...,m.shape[0] - 1) - (m.shape[0]=n*N hvis forsøget er gjort færdigt)
    avg[i] = np.mean(m[i])  #punktgennemsnit af de individuelle datamålinger
    std[i] = np.std(m[i])   #standardafvig i de individuelle datamålinger
    dosav[i] = avg[i]*K     #dosegennemsnit i hvert punkt
    dostd[i] = std[i]*K     #dosestandardafvig i hvert punkt



#### MÅLT DOSIS OG STANDARDAFVIG FRA HVERT ENKELT FORSØG ####
# if m.shape[0] == n*N:

ya = np.zeros(x_points)
yo = np.zeros(x_points)
a_matrix = np.zeros((y_lines, x_points))
o_matrix = np.zeros((y_lines, x_points))
A_matrix = np.zeros((n, y_lines, x_points))
O_matrix = np.zeros((n, y_lines, x_points))

print()
for i in range(n): #3
    for j in range(y_lines): #6
        for k in range(x_points): #5
            ya[k] = dosav[k + j*(x_points) + N*i]
            yo[k] = dostd[k + j*(x_points) + N*i]
        a_matrix[j] = ya
        o_matrix[j] = yo
    A_matrix[i] = a_matrix
    O_matrix[i] = o_matrix
# plt.rcParams.update({'font.size': 7})



#### GENNEMSNITDOSER I HVERT PUNKT OVER ALLE FORSØG OG SEM ####

DOSE_AVG = np.zeros(N)
DOSE_SEM = np.zeros(N)

for i in range(N):                                  #gå igennem alle målepunkter: range(N) = 0,1,2,...29
    point_dose_avg = np.zeros(n)                    #for hvert målepunkt; lav en 3-array med hver alle målinger i samme punkt
    for j in range(n):                              #gå igennem alle målinger i hvert punkt: range(n) = 0,1,2
        point_dose_avg[j] = dosav[i + N*j]          #fyld 3-array med alle doser målt i samme punkt
    DOSE_AVG[i] = np.mean(point_dose_avg)           #lav 30-array med gennemsnitsdosis i hvert målepunkt
    DOSE_SEM[i] = np.std(point_dose_avg)/np.sqrt(n) #lav 30-array med SEM i hvert målepunkt


y_a = np.zeros(x_points)
y_s = np.zeros(x_points)
dose_matrix = np.zeros((y_lines, x_points))
sem_matrix = np.zeros((y_lines, x_points))

for i in range(y_lines):
    for j in range(x_points):
        y_a[j] = DOSE_AVG[j + i*(x_points)]
        y_s[j] = DOSE_SEM[j + i*(x_points)]
    dose_matrix[i] = y_a
    sem_matrix[i] = y_s



#### PLOTTING ####

#intesitetsregulering i plottene så de bliver ensartede og sammenlignbare
A_min = A_matrix.min()-20
A_max = A_matrix.max()
O_min = 0#O_matrix.min()
O_max = 3.5#O_matrix.max()

#plot enkelte forsøg i samme figur
if n == 1:
    figs, axs = plt.subplots(1,2)
    axs[0].imshow(A_matrix[0], vmin=A_min, vmax=A_max, cmap='inferno',interpolation='lanczos')#, interpolation='lanczos')
    axs[0].set_title("Målt dosis (mGy)")
    axs[0].invert_yaxis()
    axs[1].imshow(O_matrix[0], vmin=O_min, vmax=O_max, cmap=plt.cm.Blues,interpolation='lanczos')#, interpolation='lanczos')
    axs[1].set_title("SD (mGy)")
    axs[1].invert_yaxis()
    for i in range(x_points):
        for j in range(y_lines):
            c = A_matrix[0,j,i]
            d = O_matrix[0,j,i]
            axs[0].text(i, j, '%.2f' %c, va='bottom', ha='center')
            axs[0].text(i, j, "±"'%.2f' %d, va='top', ha='center')
            # axs[1,i].text(j, k, '%.2f' %c, va='bottom', ha='center')
            axs[1].text(i, j, '%.2f' %d, va='center', ha='center')
else:
    figs, axs = plt.subplots(nrows=2,ncols=n,figsize=(10, 8))
    for i in range(n):
        axs[0,i].imshow(A_matrix[i], vmin=A_min, vmax=A_max, cmap='inferno',interpolation='lanczos')
        axs[0,i].set_title("Målt dosis (mGy): %s. Måling"%int(i+1))
        axs[0,i].invert_yaxis()
        axs[1,i].imshow(O_matrix[i], vmin=O_min, vmax=O_max, cmap=plt.cm.Blues,interpolation='lanczos')
        axs[1,i].set_title("SD (mGy)")
        axs[1,i].invert_yaxis()

        for j in range(x_points):
            for k in range(y_lines):
                c = A_matrix[i,k,j]
                d = O_matrix[i,k,j]
                axs[0,i].text(j, k, '%.2f' %c, va='bottom', ha='center')
                axs[0,i].text(j, k, "±"'%.2f' %d, va='top', ha='center')
                # axs[1,i].text(j, k, '%.2f' %c, va='bottom', ha='center')
                axs[1,i].text(j, k, '%.2f' %d, va='center', ha='center')
# plt.show()


#plot gennemsnitsdose og SEM fra alle forsøg
fig, ax = plt.subplots(1,2)
ax[0].imshow(dose_matrix, vmin=A_min, vmax=A_max, cmap='inferno',interpolation='lanczos')
ax[0].set_title("Gennemsnitsdosis (mGy)")
ax[0].invert_yaxis()
ax[1].imshow(sem_matrix, vmin=O_min, vmax=O_max, cmap=plt.cm.Blues,interpolation='lanczos')
ax[1].set_title("SEM (mGy)")
ax[1].invert_yaxis()
for i in range(x_points):
    for j in range(y_lines):
        c = dose_matrix[j,i]
        d = sem_matrix[j,i]
        ax[0].text(i, j, '%.2f' %c, va='bottom', ha='center')
        ax[0].text(i, j, "±"'%.2f' %d, va='top', ha='center')
        # ax[1].text(i, j, '%.2f' %c, va='bottom', ha='center')
        ax[1].text(i, j, '%.2f' %d, va='center', ha='center')
plt.show()



#### PRINT CHRG OG DOSE MÅLT I HVERT PUNKT ####
for i in range(m.shape[0]):
    if x[i]==0: #lav mellemrum i udskriften mellem hver kolonne på perspex-pladen
        print()
    print ("%s. forsøg: I feltet (%s,%s) er CHRG ="%( ( int(f[i] + 1), int(x[i]), int(y[i])  ) ),  "%.2f"%avg[i], '±', "%.2f"%std[i], "nC, og Dosen =", "%.2f"%dosav[i], "±", "%.2f"%dostd[i] , "mGy")
#### PRINT GENNEMSNITSDOSE OG SEM I HVERT MÅLEPUNKT ####
for i in range(N):
    if x[i]==0: #lav mellemrum i udskriften mellem hver kolonne på perspex-pladen
        print()
    print ("I feltet (%s,%s) er dosen i gennemsnit ="  %( int(x[i]), int(y[i]) ),"%.2f"%DOSE_AVG[i], "±","%.2f"%DOSE_SEM[i], "mGy")
print()


print()
print("antal forsøg kørt: %s " %(n))
print()
