"""
tp de reconnaissance simple
dans un premier temps on va effectuer une parametrisation
qui consiste a transformer un signal audio brut en une autre representation
plus significative
"""
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import random

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
            zcr_fenetre.append(round(calcul_zcr(fen), 1))
        fenetres.append(zcr_fenetre)
    return fenetres

def zcr_fenetrage_fichier(audio_data, pas):
    """
    calcule la zcr pour un audio fenetre
    """
    fenetres = fenetrage(audio_data, pas)
    zcr_fenetre = []
    for fen in fenetres:
        zcr_fenetre.append(round(calcul_zcr(fen), 1))
    return zcr_fenetre

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


def gaussienne(moyenne, ecart_type, taux):
    """
    dessine une gaussienne en n'utilisant pas spicy
    """
    densite = []
    distribution = np.linspace(min(taux) - 3 * ecart_type,
                    max(taux) + 3 * ecart_type, 1000)
    for tau in distribution:
        densite.append((1 / (ecart_type * math.sqrt(2 * math.pi)))
                       * math.exp(-((tau - moyenne)**2) / (2 * ecart_type**2)))
    plt.hist(taux, bins=10, density=True, alpha=0.5, label='Données de taux')
    plt.plot(distribution, densite, 'r', label='Gaussienne')
    plt.show()

def expection_maximisation(M, moy, ecart ):
    """
    calcule la vraisemblance
    """
    #partie initialisation
    mean_hasard = []
    intervalle = [moy-ecart, moy+ecart]
    for i in range(M):
        mean_hasard.append(random.uniform(moy-ecart, moy+ecart))
    #partie expectation
    #partie maximisation
    return mean_hasard
    

def gaussiennes_multiple(moyennes, ecarts_types, poids, taux):
    """
    Dessine plusieurs gaussiennes en utilisant les paramètres fournis.
    """
    plt.hist(taux, bins=20, density=True, alpha=0.5, label='Données de taux', color='blue')
    for moyenne, ecart_type, poids_component in zip(moyennes, ecarts_types, poids):
        distribution = np.linspace(min(taux) - 3 * ecart_type, max(taux) + 3 * ecart_type, 1000)
        densite = (1 / (ecart_type * math.sqrt(2 * math.pi))) * np.exp(-((distribution - moyenne)**2) / (2 * ecart_type**2))
        #densite_totale += dentiste * poids_component
        plt.plot(distribution, densite, label=f'Gaussienne {moyenne:.2f}', linestyle='--')
    
    plt.xlabel('Taux')
    plt.ylabel('Densité de Probabilité')
    plt.title('Gaussiennes Multiples')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    audio_da = ouverture_fichier_audio("data/0.raw")
    fenetres = zcr_fenetrage_fichier(audio_da, int(sys.argv[1]))
    #print(fenetres)
    ZCRSS = fenetres[0:len(fenetres)//2]
    ZCRSSS = fenetres[len(fenetres)//2:]
    moy, ecart = moyenne_ecarttype(ZCRSS)
    moyy, ecartt = moyenne_ecarttype(ZCRSSS)
    m = [moy , moyy] 
    e =[ ecart , ecartt]
    t = ZCRSS + ZCRSSS
    print(expection_maximisation(3, moy, ecart))
    #gaussiennes_multiple(m, e, [0.5, 0.5], t)
