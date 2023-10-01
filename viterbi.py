# -*- coding: utf-8 -*-
"""
implente l algorithme de viterbi pour determiner un mot a partir
d un vecteur d observation
"""
import numpy as np


def viterbi(observation, proba_init, transition, emission):
    """
    parametres :
        observation : matrice d observation
        proba_init : probabilites initiales
        transition : matrice de transition
        emission : matrice d emission
    retour :

    """
    len_o = len(observation)  # T pour le temps
    len_init = len(proba_init)
    proba = np.zeros((len_init, len_o))
    chemin = np.zeros((len_init, len_o), dtype=int)

    # Initialisation de la première colonne de delta
    # initilaisier la premiere colonne de delta
    # elle prend la probabilité initiale pi[i] et la multiplie par emission[:, observation[0]]
    # qui est la probabilité d'observer observation[0] dans cet état
    proba[:, 0] = proba_init * emission[:, observation[0]]

    # Récursion : calcul des probabilites maximale a chaqye etape
    for ite in range(1, len_o):
        for j in range(len_init):
            temp = np.zeros(len_init)
            for i in range(len_init):
                temp[i] = proba[i, ite - 1] * transition[i, j] * \
                    emission[j, observation[ite]]
            proba[j, ite] = np.max(temp)
            chemin[j, ite] = np.argmax(temp)

    # Terminaison
    best_prob = np.max(proba[:, -1])
    best_path = [np.argmax(proba[:, -1])]

    # Backtracking
    # va jusqu a T-2 car T-1 est le dernier etat, c est evident d y arriver
    for i in range(len_o - 2, -1, -1):
        # revient en arriere dans le temps en suivant le chemin
        # cela nous indique comment le modèle estime que nous sommes passés
        # de l'état best_path[-1] à l'état suivant à l'instant t + 1.
        best_path.append(chemin[best_path[-1], i + 1])

    best_path.reverse()
    print(best_prob)

    return best_prob


pi_oui = np.array([1, 0, 0])
a_oui = np.array([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.0, 0.5, 0.5]])
b_oui = np.array([[2, 4, 3], [2, 4, 3], [2, 4, 3]])

pi_non = np.array([1, 0, 0])
a_non = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
b_non = np.array([[3, 2, 1], [3, 2, 1], [3, 2, 1]])

observations = np.array([0, 2, 2])

# Application de l'algorithme de Viterbi aux deux modèles
best_prob_oui = viterbi(observations, pi_oui, a_oui, b_oui)
# best_prob_non, best_path_non = viterbi(observation, pi_non, a_non, b_non)
