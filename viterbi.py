# -*- coding: utf-8 -*-
"""
implente l algorithme de viterbi pour determiner un mot a partir
d un vecteur d observation
"""
import numpy as np

def viterbi(observation, pi, transition, emission):
    """
    parametres :
        observation : matrice d observation
        pi : probabilites initiales
        transition : matrice de transition
        emission : matrice d emission
    retour :
        
    """
    T = len(observation)# T pour le temps 
    N = len(pi)
    proba = np.zeros((N, T)) 
    chemin = np.zeros((N, T), dtype=int) 

    # Initialisation de la première colonne de delta
    #initilaisier la premiere colonne de delta
    #elle prend la probabilité initiale pi[i] et la multiplie par emission[:, observation[0]]
    #qui est la probabilité d'observer observation[0] dans cet état
    proba[:, 0] = pi * emission[:, observation[0]]

    # Récursion : calcul des probabilites maximale a chaqye etape 
    for t in range(1, T):
        for j in range(N):
            temp = np.zeros(N)
            for i in range(N):
                temp[i] = proba[i, t - 1] * transition[i, j] * emission[j, observation[t]]
            proba[j, t] = np.max(temp)
            chemin[j, t] = np.argmax(temp)

    # Terminaison
    best_prob = np.max(proba[:, -1])
    best_path = [np.argmax(proba[:, -1])]
    
    # Backtracking
    #va jusqu a T-2 car T-1 est le dernier etat, c est evident d y arriver
    for t in range(T - 2, -1, -1):
        #revient en arriere dans le temps en suivant le chemin
        #cela nous indique comment le modèle estime que nous sommes passés
        #de l'état best_path[-1] à l'état suivant à l'instant t + 1.
        best_path.append(chemin[best_path[-1], t + 1])

    best_path.reverse()
    print(best_prob)

    return best_prob

pi_oui = np.array([1, 0, 0])
a_oui = np.array([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.0, 0.5, 0.5]])
b_oui = np.array([[2, 4, 3], [2, 4, 3], [2, 4, 3]])

pi_non = np.array([1, 0, 0])
a_non = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
b_non = np.array([[3, 2, 1], [3, 2, 1], [3, 2, 1]])

observation = np.array([0, 2, 2])

# Application de l'algorithme de Viterbi aux deux modèles
best_prob_oui = viterbi(observation, pi_oui, a_oui, b_oui)
#best_prob_non, best_path_non = viterbi(observation, pi_non, a_non, b_non)
