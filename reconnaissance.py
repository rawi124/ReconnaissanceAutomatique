"""
tp de reconnaissance simple
dans un premier temps on va effectuer une parametrisation
qui consiste a transformer un signal audio brut en une autre representation
plus significative
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def ouverture_fichier_audio(fichier):
    """
    effectue l'ouverture d'un fichier audio et extrait son contenu
    """
    audio_data = np.fromfile(fichier, dtype=np.int16)
    return audio_data


def calcul_zcr(audio_data):
    """
    calcul le taux de passage par zero d un signal
    en effet si une valeur est positive et que sa precedente est negative
    ( et l inverse ) alors un passage par zero a eu lieu
    """
    zcr_audio = 0
    for i in range(1, len(audio_data)):
        if audio_data[i-1] * audio_data[i] < 0:
            zcr_audio += 1
    return zcr_audio/len(audio_data)


def zcr_tous_fichiers(chemin, pas):
    """
    calcule la zcr pour chaque fenetre selon un certain pas
    pour chaque fichier dans le repertoire en parametre
    retourne une liste de liste de taux
    """
    fenetres = []
    lettres = "0123456789abcdefghijklmnopqrsuvwxyz"
    for lettre in lettres:
        fichier = chemin + "/" + lettre + ".raw"
        audio_data = ouverture_fichier_audio(fichier)
        fenetre = fenetrage(audio_data, pas)
        zcr_fenetre = []
        for fen in fenetre:
            zcr_fenetre.append(round(calcul_zcr(fen), 2))
        fenetres.append(zcr_fenetre)
    return fenetres


def fenetrage(audio_data, pas):
    """
    effectue le fenetrage de l audio selon un certain pas
    """
    fenetres = []
    i = 0
    while i+pas < len(audio_data):
        fenetres.append(audio_data[i:i+pas])
        i += pas
    fenetres.append(audio_data[i:])
    return fenetres


def moyenne_ecarttype(taux_fenetre):
    """
    calcule la moyenne et l ecart type
    """
    return np.mean(taux_fenetre), np.std(taux_fenetre)


def moyenne_ecarttype_tous(zcr_fichiers):
    """
    calcule la moyenne et l ecart type pour chaque fichier
    """
    gmm = []
    for zcr_fichier in zcr_fichiers:
        gmm.append(moyenne_ecarttype(zcr_fichier))
    return gmm

def affichage_gaussienne(moyenne, ecart_type, taux):
    """
    affiche la gaussienne et l histogramme des taux
    """
    distribution = np.linspace(min(taux) - 0.2, max(taux) + 0.2, 1000)
    gaussian = norm.pdf(distribution, moyenne, ecart_type)
    plt.hist(taux, bins=10, density=True, alpha=0.5, label='Données de taux')
    plt.plot(distribution, gaussian, 'r', label='Gaussienne')
    plt.xlabel('Taux')
    plt.ylabel('Densité de Probabilité')
    plt.title('Données de Taux et Distribution Gaussienne pour le fichier 0')
    #img_fich = "data/img_fich0.png"
    #plt.savefig(img_fich)
    plt.show()

if __name__ == "__main__":
    audio_da = ouverture_fichier_audio("data/0.raw")
    ZCR = calcul_zcr(audio_da)
    ZCRS = zcr_tous_fichiers("data", int(sys.argv[1]))
    moy, ecart = moyenne_ecarttype(ZCRS[int(sys.argv[2])])
    affichage_gaussienne(moy, ecart, ZCRS[int(sys.argv[2])])
    # print(etiquettage(ZCRS))
