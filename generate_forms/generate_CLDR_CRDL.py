import numpy as np
import matplotlib.pyplot as plt


def generate_CLDR(params: dict):
    """
    Calcule les valeurs d'une fonction définie par morceaux sur un intervalle [a, b],
    avec une croissance exponentielle controlée entre a et c, et une décroissance linéaire de c à b.

    La fonction est définie comme suit :
    - Pour x dans [a, c[ (c non inclus), y =  exp(A*(x-a))-1, où A est l'amplitude de croissance.
    - Pour x dans [c, b], la fonction est linéaire et décroît jusqu'à atteindre 0 au point b.
      La pente et l'ordonnée à l'origine de cette portion linéaire sont calculées pour assurer une
      transition douce en c, de sorte que la fonction soit continue.

    Paramètres :
    - params : dict, un dictionnaire contenant les paramètres de la fonction.
        - start (int): Indice de départ de la fonction (peut être négatif).
        - width (int): Largeur de la fonction en nombre de points.
        - middle (float): Ratio de la largeur de la croissance exponentielle par rapport à la largeur totale. Doit être compris entre 0.5 et 1.
        - A (float): Amplitude de la croissance exponentielle.

    Retour :
    - y : array_like, les valeurs de la fonction évaluées à chaque point x.
    """

    a = params['start']
    b = params['width']
    c = int(params['middle']*params['width'])
    A = params['A']

    y = np.zeros(b)

    # Première partie de la fonction : croissance exponentielle
    y[max(a,0):min(c+a,b)] = np.exp(A*(np.arange(max(a,0), min(c+a,b), 1)-a)) - 1
    
    # Deuxième partie de la fonction : décroissance linéaire
    k = ((b+a)*(np.exp(A*(c)) - 1) )/(b-c)
    m = -k/(b+a)
    y[c+a:min(b+a, b)] = m * (np.arange(c+a, min(b+a, b), 1)) + k

    # Assurer que y a la hauteur demandée
    if y.max() == 0:
        print("La fonction est nulle sur tout l'intervalle, vérifiez les paramètres.")
        print(params)
    return y/y.max()*params['height']


def generate_CRDL(params: dict):
    # Un CRDL est juste le symétrique d'un CLDR.
    # On peut donc simplement inverser les valeurs de la fonction CLDR, en utilisant -start pour avoir un décalage dans le bon sens
    return generate_CLDR({**params, 'start':-params['start']})[::-1] 


if __name__ == "__main__":
    # c= 0.18-0.24*width
    width=500 # width=200 -- height=1-3 ; width=100 -- height= 12-14 ; width=300 -- height=0.3-0.6 ; width=500 -- height=0.01-0.03
    height=40
    
    params= {
        'start': 0,
        'width': 500,
        'middle': 0.75,
        'A': 0.03,
        'height': 40
    }

    y_cldr_A_0_01 = generate_CLDR({**params, 'A': 0.01})
    y_cldr_A_0_02 = generate_CLDR({**params, 'A': 0.02})
    y_cldr_A_0_03 = generate_CLDR({**params, 'A': 0.03})
    y_cldr_A_0_04 = generate_CLDR({**params, 'A': 0.04})
    y_cldr_A_0_05 = generate_CLDR({**params, 'A': 0.05})
    #y_crdl = generate_CRDL(params1)
    plt.figure(figsize=(13, 7))
    plt.rc('legend',fontsize=7)

    plt.subplot(3, 2, 1)
    plt.plot(generate_CLDR({**params, 'A': 0.01}), label="CLDR - A=0.01", color='darkred')
    plt.plot(generate_CLDR({**params, 'A': 0.02}), label="CLDR - A=0.02", color='red')
    plt.plot(generate_CLDR({**params, 'A': 0.03}), label="CLDR - A=0.03", color='green')
    plt.plot(generate_CLDR({**params, 'A': 0.04}), label="CLDR - A=0.04", color='blue')
    plt.plot(generate_CLDR({**params, 'A': 0.05}), label="CLDR - A=0.05", color='darkblue')
    plt.grid()
    plt.legend()
    plt.title("Influence de A sur CLDR")

    plt.subplot(3, 2, 2)
    plt.plot(generate_CRDL({**params, 'start': -100}), label="CRDL - Start=-100", color='darkred')
    plt.plot(generate_CRDL({**params, 'start': -50}), label="CRDL - Start=-50", color='red')
    plt.plot(generate_CRDL({**params, 'start': 0}), label="CRDL - Start=0", color='green')
    plt.plot(generate_CRDL({**params, 'start': 50}), label="CRDL - Start=50", color='blue')
    plt.plot(generate_CRDL({**params, 'start': 100}), label="CRDL - Start=100", color='darkblue')
    plt.grid()
    plt.legend()
    plt.title("Influence de Start sur CRDL")

    plt.subplot(3, 2, 3)
    plt.plot(generate_CLDR({**params, 'middle': 0.55}), label="CLDR - Middle=0.55", color='red')
    plt.plot(generate_CLDR({**params, 'middle': 0.75}), label="CLDR - Middle=0.75", color='green')
    plt.plot(generate_CLDR({**params, 'middle': 0.95}), label="CLDR - Middle=0.95", color='blue')

    plt.grid()
    plt.legend()
    plt.title("Influence de Middle sur CLDR")

    plt.subplot(3, 2, 4)
    plt.plot(generate_CRDL({**params, 'width': 100}), label="CRDL - Width=100", color='darkred')
    plt.plot(generate_CRDL({**params, 'width': 200}), label="CRDL - Width=200", color='red')
    plt.plot(generate_CRDL({**params, 'width': 300}), label="CRDL - Width=300", color='green')
    plt.plot(generate_CRDL({**params, 'width': 400}), label="CRDL - Width=400", color='blue')
    plt.plot(generate_CRDL({**params, 'width': 500}), label="CRDL - Width=500", color='darkblue')
    plt.grid()
    plt.legend()
    plt.title("Influence de Width sur CRDL")

    plt.subplot(3, 2, 5)
    plt.plot(generate_CLDR({**params, 'height': 20}), label="CLDR - Height=20", color='darkred')
    plt.plot(generate_CLDR({**params, 'height': 30}), label="CLDR - Height=30", color='red')
    plt.plot(generate_CLDR({**params, 'height': 40}), label="CLDR - Height=40", color='green')
    plt.plot(generate_CLDR({**params, 'height': 50}), label="CLDR - Height=50", color='blue')
    plt.plot(generate_CLDR({**params, 'height': 60}), label="CLDR - Height=60", color='darkblue')
    plt.grid()
    plt.legend()
    plt.title("Influence de Height sur CLDR")

    plt.tight_layout()
    plt.show()
