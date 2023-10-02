"""
tp de reconnaissance simple
dans un premier temps on va effectuer une parametrisation
qui consiste a transformer un signal audio brut en une autre representation
plus significative
"""


def ouverture_fichier_audio(fichier):
    """
    effectue l'ouverture d'un fichier audio et extrait son contenu
    """
    with open(fichier, "rb") as file:
        audio_data = file.read()
    audio_data = list(audio_data)
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
    return zcr_audio
def zcr_tous_fichiers(chemin):
    """
    calcule la zcr pour chaque fichier dans le repertoire en parametre
    """
    for i in range(10):
        fichier = chemin + "/" + str(i) + ".raw"
        print(fichier)
        audio_data = ouverture_fichier_audio(fichier)
        print(calcul_zcr(audio_data))
    for i in ['a', 'b', 'c' ] :
        fichier = chemin + "/" + i + ".raw"
        print(fichier)
        audio_data = ouverture_fichier_audio(fichier)
        print(calcul_zcr(audio_data))


if __name__ == "__main__":
    audio_da = ouverture_fichier_audio("data/0.raw")
    ZCR = calcul_zcr(audio_da)
    print(ZCR)
    zcr_tous_fichiers("data")
