"""
tp de reconnaissance simple
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


if __name__ == "__main__":
    audio_da = ouverture_fichier_audio("data/0.raw")
    ZCR = calcul_zcr(audio_da)
    print(ZCR)
