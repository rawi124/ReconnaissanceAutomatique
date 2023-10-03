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
    audio_data = np.fromfile(fichier)
    return audio_data


def calcul_zcr(audio_data):
    """
    calcul le taux de passage par zero d un signal
    en effet si une valeur est positive et que sa precedente est negative
    ( et l inverse ) alors un passage par zero a eu lieu
    """
    zcr_audio = 0
    for i in range(1, len(audio_data)):
        # print(audio_data[i])
        if np.float64(audio_data[i-1]) * np.float64(audio_data[i]) < 0:
            zcr_audio += 1
    return zcr_audio/len(audio_data)


def zcr_tous_fichiers(chemin):
    """
    calcule la zcr pour chaque fichier dans le repertoire en parametre
    """
    zcr_s = []
    lettres = "0123456789abcdefghijklmnopqrsuvwxyz"
    for lettre in lettres:
        fichier = chemin + "/" + lettre + ".raw"
        audio_data = ouverture_fichier_audio(fichier)
        zcr_s.append(calcul_zcr(audio_data))
    zcr_s_arrondies = [round(zcr, 1) for zcr in zcr_s]
    return zcr_s_arrondies


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
    ZCRS = zcr_tous_fichiers("data")
    print(etiquettage(ZCRS))
