import numpy as np
import matplotlib.pyplot as plt

scope = np.array([0.1, 0.2, 0.3, 0.5, 1.0])

M15 = np.array([62, 72, 58, 60, 68])
M16 = np.array([14, 16, 19, 19, 18])
M17 = np.array([51, 52, 54, 55, 61])
M18 = np.array([16, 16, 18, 18, 21])
M15p = np.array([76, 74, 66, 70, 64])
M16p = np.array([12, 13, 12, 13, 11])
M17p = np.array([67, 69, 70, 70, 70])
M18p = np.array([10, 10, 11, 10, 11])

fig, ax = plt.subplots(2,2, figsize=(14,8))

ax[0,0].plot(scope, M15 ,"b", label="bestrålet")
ax[0,0].plot(scope, M17 ,"r", label="kontrol")
ax[0,0].plot(scope, M15p,"g", label="bestrålet, primet")
ax[0,0].plot(scope, M17p,"y", label="kontrol, primet")
ax[0,0].axhline(np.mean(M15 ), color="b",linestyle="--")
ax[0,0].axhline(np.mean(M17 ), color="r",linestyle="--")
ax[0,0].axhline(np.mean(M15p), color="g",linestyle="--")
ax[0,0].axhline(np.mean(M17p), color="y",linestyle="--")
ax[0,0].set_title("Celler i G2-arrest")
ax[0,0].set_xlabel("Dose (Gy)")
ax[0,0].set_ylabel("Andel ikke i mitose (%)")
ax[0,0].legend()
ax[0,0].set_xticks(scope)
ax[1,0].plot(scope, M16 ,"b", label="bestrålet")
ax[1,0].plot(scope, M18 ,"r", label="kontrol")
ax[1,0].plot(scope, M16p,"g", label="bestrålet, primet")
ax[1,0].plot(scope, M18p,"y", label="kontrol, primet")
ax[1,0].axhline(np.mean(M16 ), color="b",linestyle="--")
ax[1,0].axhline(np.mean(M18 ), color="r",linestyle="--")
ax[1,0].axhline(np.mean(M16p), color="g",linestyle="--")
ax[1,0].axhline(np.mean(M18p), color="y",linestyle="--")
ax[1,0].set_title("Celler i mitose")
ax[1,0].set_xlabel("Dose (Gy)")
ax[1,0].set_ylabel("Andel i mitose (%)")
ax[1,0].legend()
ax[1,0].set_xticks(scope)
#
# ax[0,1].plot(scope, M15/np.mean(M15),"b", label="bestrålet, ej mitose")
# ax[0,1].plot(scope, M16/np.mean(M16),"r", label="bestrålet, mitose")
# ax[0,1].plot(scope, M17/np.mean(M17),"g", label="kontrol, ej mitose")
# ax[0,1].plot(scope, M18/np.mean(M18),"y", label="kontrol, mitose")
# ax[0,1].axhline(np.mean(M15)/np.mean(np.array([M15, M16,M17,M18])), color="b",linestyle="--")
# ax[0,1].axhline(np.mean(M16)/np.mean(np.array([M15, M16,M17,M18])), color="r",linestyle="--")
# ax[0,1].axhline(np.mean(M17)/np.mean(np.array([M15, M16,M17,M18])), color="g",linestyle="--")
# ax[0,1].axhline(np.mean(M18)/np.mean(np.array([M15, M16,M17,M18])), color="y",linestyle="--")
# ax[0,1].set_title("Trends, uprimede celler")
# ax[0,1].set_xlabel("Dose (Gy)")
# ax[0,1].set_ylabel("Normaliseret")
# ax[0,1].legend()
# ax[0,1].set_xticks(scope)
# ax[1,1].plot(scope, M15p/np.mean(M15p),"b", label="bestrålet, primet, ej mitose")
# ax[1,1].plot(scope, M16p/np.mean(M16p),"r", label="bestrålet, primet, mitose")
# ax[1,1].plot(scope, M17p/np.mean(M17p),"g", label="kontrol, primet, ej mitose")
# ax[1,1].plot(scope, M18p/np.mean(M18p),"y", label="kontrol, primet, mitose")
# ax[1,1].axhline(np.mean(M15p)/np.mean(np.array([M15p, M16p,M17p,M18p])), color="b",linestyle="--")
# ax[1,1].axhline(np.mean(M16p)/np.mean(np.array([M15p, M16p,M17p,M18p])), color="r",linestyle="--")
# ax[1,1].axhline(np.mean(M17p)/np.mean(np.array([M15p, M16p,M17p,M18p])), color="g",linestyle="--")
# ax[1,1].axhline(np.mean(M18p)/np.mean(np.array([M15p, M16p,M17p,M18p])), color="y",linestyle="--")
# ax[1,1].set_title("Trends, primede celler")
# ax[1,1].set_xlabel("Dose (Gy)")
# ax[1,1].set_ylabel("Normaliseret")
# ax[1,1].legend()
# ax[1,1].set_xticks(scope)
#
# ax[1,2].plot(M15/np.mean(M15), M15p/np.mean(M15p),"*", label="M15")
# ax[1,2].plot(M16/np.mean(M16), M16p/np.mean(M16p),"*", label="M16")
# ax[1,2].plot(M17/np.mean(M17), M17p/np.mean(M17p),"*", label="M17")
# ax[1,2].plot(M18/np.mean(M18), M18p/np.mean(M18p),"*", label="M18")
# ax[1,2].set_title("Trends")
# ax[1,2].set_xlabel("Uprimede celler")
# ax[1,2].set_ylabel("Primede celler")
# ax[1,2].legend()

plt.tight_layout()
plt.show()
