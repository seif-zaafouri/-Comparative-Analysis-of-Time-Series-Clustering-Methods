import matplotlib.pyplot as plt
import numpy as np


def generate_M(params: dict):
    """
    Generate M-shaped values based on the input parameters.

    Args:
        n (int): Number of points to generate.
        params (dict): Dictionary containing the parameters for generating the M shape.
            - 'start' (int): starture value.
            - 'width' (int): Width of the M shape.
            - 'height' (int): Height of the M shape.

    Returns:
        list: List of M-shaped values corresponding to the input values.

    """
    start = params['start']
    width = params['width']
    height = params['height']

    y = np.zeros(width)
    
    for i in range(width):
        if (i-start) < (width/4) and i >= start:
            y[i] = (4*height*i)/(width) - (4*height*start)/(width)
        elif (i-start) < (width/2) and (i-start) >= (width/4):
            y[i] = (-2*height*i)/(width) + height/2 + (2*height*(start + width/2))/width
        elif (i-start) >= (width/2) and (i-start) < ((3*width)/4):
            y[i] = (2*height*i)/(width) + height/2 - (2*height*(start + width/2))/width 
        elif (i-start) >= ((3*width)/4) and (i-start) < width:
            y[i] = -(4*height*i)/(width) + (4*height*(start+width))/(width)
        else:
            y[i] = 0
    return y


if __name__ == "__main__":
    n = 100
    params = {'start': 40, 'width': 50, 'height': 50}
    y = generate_M(n, params)
    plt.plot(y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Uppercase M')
    plt.grid(True)
    plt.show()

    
