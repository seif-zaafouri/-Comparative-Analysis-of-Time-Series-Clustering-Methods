import numpy as np
import matplotlib.pyplot as plt

def generate_CLDR(n: int, params: dict):
    """
    Calcule les valeurs d'une fonction définie par morceaux sur un intervalle [a, b],
    avec une croissance exponentielle controlée entre a et c, et une décroissance linéaire de c à b.

    La fonction est définie comme suit :
    - Pour x dans [a, c[ (c non inclus), y =  exp(A*(x-a))-1, où A est l'amplitude de croissance.
    - Pour x dans [c, b], la fonction est linéaire et décroît jusqu'à atteindre D au point b.
      La pente et l'ordonnée à l'origine de cette portion linéaire sont calculées pour assurer une
      transition douce en c, de sorte que la fonction soit continue.

    Paramètres :
    - n : int, le nombre de points à générer dans l'intervalle [a, b].
    - params : dict, un dictionnaire contenant les paramètres de la fonction.
        - a : float, borne inférieure de l'intervalle de définition de la fonction.
        - b : float, borne supérieure de l'intervalle de définition de la fonction.
        - c : float, point de coupure entre la croissance en exponentielle et la décroissance linéaire.
        - A : float, amplitude de la partie croissance en exponentielle.
        - D : float, valeur finale de la fonction à x = b, doit être inférieure ou égale à f(c) et supérieure ou égale à 0.

    Retour :
    - x : array_like, les valeurs de x générées.
    - y : array_like, les valeurs de la fonction évaluées à chaque point x.
    """
    a = params['a']
    b = params['b']
    c = params['c']
    A = params['A']
    D = params['D']

    # Génération de x sur l'intervalle [a, b] avec n points
    x = np.linspace(a, b, n)

    y = np.zeros_like(x)
    y[(x < c) & (x >= a)] = np.exp(A*(x[(x < c) & (x >= a)]-a)) - 1
    k = (b*(np.exp(A*(c-a)) - 1) - D*c)/(b-c)
    m = (D-k)/b
    y[(x <= b) & (x >= c)] = m * x[(x <= b) & (x >= c)] + k
    return y


def generate_CRDL(n: int, params: dict):
    """
    Calcule les valeurs d'une fonction définie par morceaux sur un intervalle [a, b],
    avec une croissance linéaire entre a et c, suivie d'une décroissance exponentielle contrôlée de c à b.

    Paramètres :
    - n : int, le nombre de points à générer dans l'intervalle [a, b].
    - params : dict, un dictionnaire contenant les paramètres de la fonction.
        - a : float, borne inférieure de l'intervalle de définition de la fonction.
        - b : float, borne supérieure de l'intervalle de définition de la fonction.
        - c : float, point où la croissance linéaire s'arrête et où la décroissance exponentielle commence.
        - A : float, une constante qui contrôle le taux de décroissance de la partie exponentielle.
        - D : float, la valeur cible de la fonction lorsque `x = b`, indiquant le niveau final après la décroissance.

    Retour :
    - x : array_like, les valeurs de x générées.
    - y : array_like, les valeurs de la fonction calculées pour chaque point x.
    """
    a = params['a']
    b = params['b']
    c = params['c']
    A = params['A']
    D = params['D']

    # Génération de x sur l'intervalle [a, b] avec n points
    x = np.linspace(a, b, n)

    y = np.zeros_like(x)
    m = (np.exp(-A*(c-b))-1+D)/(c-a)
    k = -m*a
    y[(x < c) & (x >= a)] = m * x[(x < c) & (x >= a)] + k
    y[(x <= b) & (x >= c)] = np.exp(-A*(x[(x <= b) & (x >= c)]-b)) - 1 + D
    return y





if __name__ == "__main__":
    n = 100  # Nombre de points à générer

    params = {
        'a': 0,
        'b': n*0.1,
        'c': 6.0,
        'A': 0.2,
        'D': 1
    }
    

    y1 = generate_CLDR(n, params)
    y2 = generate_CRDL(n-20, params)

    plt.plot(y1, label='CLDR')
    plt.plot(y2, label='CRDL')
    plt.legend()

    plt.show()