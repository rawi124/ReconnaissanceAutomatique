"""
tp de reconnaissance simple
dans un premier temps on va effectuer une parametrisation
qui consiste a transformer un signal audio brut en une autre representation
plus significative
"""
import numpy as np


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
    calcule la zcr pour chaque fichier dans le repertoire en parametre
    """
    fenetres = {}
    lettres = "0123456789abcdefghijklmnopqrsuvwxyz"
    for lettre in lettres:
        fichier = chemin + "/" + lettre + ".raw"
        audio_data = ouverture_fichier_audio(fichier)
        fenetre = fenetrage(audio_data, pas)
        zcr_fenetre = []
        for fen in fenetre :
            zcr_fenetre.append(round(calcul_zcr(fen), 2))
        fenetres[fichier] = zcr_fenetre
    return fenetres

def fenetrage(audio_data, pas):
    """
    effectue le fenetrage de l audio selon un certain pas
    """
    fenetres = []
    i = 0
    while i+pas < len(audio_data) :
        fenetres.append(audio_data[i:i+pas])
        i += pas
    fenetres.append(audio_data[i:])
    return fenetres 

def moyenne_ecarttype(taux_fenetre):
    """
    calcule la moyenne et l ecart type 
    """
    return np.mean(taux_fentre), np.std(taux_fenetre)

def moyenne_ecarttype_tous(zcr_tous_fichiers):
    """
    calcule la moyenne et l ecart type pour chaque fichier
    """
    gmm = []
    for el in zcr_tous_fichiers.values():
        gmm.append(moyenne_ecarttype(el))
    return gmm

def etiquettage(zcr_s):
    """
    effectue ettiquetage en gardant que les taux 
    de passage par zero qui ont plus de deux occurences
    """
    occurences = {}
    for occ in zcr_s:
        if occ in occurences:
            occurences[occ] += 1
        else:
            occurences[occ] = 1
    return {element: count for element, count in occurences.items() if count > 1}


if __name__ == "__main__":
    audio_da = ouverture_fichier_audio("data/0.raw")
    ZCR = calcul_zcr(audio_da)
    #print(zcr_tous_fichiers("data", 551))
    print(moyenne_ecarttype_tous)
    #print(etiquettage(ZCRS))
