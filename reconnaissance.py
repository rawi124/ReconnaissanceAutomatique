"""
tp de reconnaissance simple
dans un premier temps on va effectuer une parametrisation
qui consiste a transformer un signal audio brut en une autre representation
plus significative
"""
import sys
import math
import random
import numpy as np
import matplotlib.pyplot as plt


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


def gaussienne_plot(moyenne, ecart_type, taux):
    """
    dessine une gaussienne en n'utilisant pas spicy
    """
    densite = gaussienne(moyenne, ecart_type, taux)
    distribution = np.linspace(min(taux) - 3 * ecart_type,
                               max(taux) + 3 * ecart_type, 1000)
    plt.hist(taux, bins=10, density=True, alpha=0.5, label='Données de taux')
    plt.plot(distribution, densite, 'r', label='Gaussienne')
    plt.show()


def gaussienne(moyenne, ecart_type, taux):
    """
    calcule la gaussienne
    """
    densite = []
    distribution = np.linspace(min(taux) - 3 * ecart_type,
                               max(taux) + 3 * ecart_type, 1000)
    for tau in distribution:
        densite.append((1 / (ecart_type * math.sqrt(2 * math.pi)))
                       * math.exp(-((tau - moyenne)**2) / (2 * ecart_type**2)))
    return densite


def expectation_maximisation(taux, nb_gauss, moy, ecart, seuil):
    """
    Calcule la vraisemblance en utilisant l'algorithme Expectation-Maximization.
    """
    mean_hasard = [random.uniform(moy * 0.9, moy * 1.1)
                   for _ in range(nb_gauss)]
    ecart_hasard = [random.uniform(ecart * 0.9, ecart * 1.1)
                    for _ in range(nb_gauss)]
    poids = [1 / nb_gauss] * nb_gauss

    prev_log_likelihood = float('-inf')

    while True:
        len_taux = len(taux)
        probabilites = np.zeros((len_taux, nb_gauss))

        for j in range(nb_gauss):
            probabilites[:, j] = poids[j] * 1 / ((np.sqrt(2 * np.pi)) * ecart_hasard[j]) * \
                np.exp(-0.5 * ((taux - mean_hasard[j]) / ecart_hasard[j]) ** 2)
            probabilites = probabilites / \
                np.sum(probabilites, axis=1, keepdims=True)

        log_likelihood_value = log_likelihood(
            taux, mean_hasard, ecart_hasard, poids)
        print(f"Log-likelihood = {log_likelihood_value}")

        if log_likelihood_value - prev_log_likelihood < seuil:
            print("Convergence atteinte. Arrêt de l'itération.")
            break

        prev_log_likelihood = log_likelihood_value

        for i in range(nb_gauss):
            mean_hasard[i] = np.sum(
                probabilites[:, i] * taux) / np.sum(probabilites[:, i])
            ecart_hasard[i] = np.sqrt(np.sum(
                probabilites[:, i] * (taux - mean_hasard[i]) ** 2) / np.sum(probabilites[:, i]))
            poids[i] = np.mean(probabilites[:, i])

    return mean_hasard, ecart_hasard, poids

def plot_gaussians(data, moy_em, et_em):
    """
    dessine un melange de gaussienne
    """
    moy_em = np.array(moy_em)
    et_em = np.array(et_em)
    distribution = np.linspace(
        min(moy_em - 3 * et_em), max(moy_em + 3 * et_em), 1000)
    for j, moy in enumerate(moy_em):
        gaussian = (1 / (np.sqrt(2 * np.pi) * et_em[j])) *\
            np.exp(-0.5 * ((distribution - moy) / et_em[j]) ** 2)
        plt.plot(distribution, gaussian, label=f'Composante {j+1}')

    plt.hist(data, bins=10, density=True, alpha=0.5, label='Données')
    plt.xlabel('Valeur')
    plt.ylabel('Densité de probabilité')
    plt.legend()
    plt.show()


def log_likelihood(zcr_taux, moyenne, ecart, poids):
    """
    Calcule le log-vraisemblance pour un ensemble de données en utilisant un modèle GMM.
    """
    log_ll_total = 0
    for tau in zcr_taux:
        ll_x = 0
        for j, moyen in enumerate(moyenne):
            prob = (1 / (np.sqrt(2 * np.pi) *
                    ecart[j])) * np.exp(-0.5 * ((tau - moyen) / ecart[j]) ** 2)
            ll_x += poids[j] * prob
        log_ll_total += np.log(ll_x)
    return log_ll_total


def matrice_ll(chemin, pas, nb_gauss, seuil):
    """
    renvoie une matrice de tous les log_lik
    """
    zcr_tous = zcr_tous_fichiers(chemin, pas)
    matrice = []
    for tau in zcr_tous:
        moyenn, ecar = moyenne_ecarttype(tau)
        moyenn, ecar, poid = expectation_maximisation(tau, nb_gauss, moyenn, ecar, seuil)
        matrice.append(log_likelihood(tau, moyenn, ecar, poid))
    return matrice


if __name__ == "__main__":
    AUDIO = ouverture_fichier_audio("data/c.raw")
    FENET = zcr_fenetrage_fichier(AUDIO, int(sys.argv[1]))
    MO, ECAR = moyenne_ecarttype(FENET)
    # MOH, ECH, POIDSH = expectation_maximisation(FENET, 2, MO, ECAR, 20)
    # plot_gaussians(FENET, MOH, ECH)
    print(matrice_ll("data", int(sys.argv[1]), 2, 100))
