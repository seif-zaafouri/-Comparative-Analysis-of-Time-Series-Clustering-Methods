

import matplotlib.pyplot as plt
import numpy as np


def generate_parabola(params: dict):
    """
    Generates a parabola waveform with a given start, width, and height.

    Args:
        n (int): The number of points to generate.
        params (dict): A dictionary containing the parameters for generating the parabola waveform.
            - start (int): The starting index of the parabola waveform.
            - width (int): The duration (width) of the parabola waveform.
            - height (float): The peak height of the parabola waveform.

    Returns:
        np.ndarray[float]: The generated parabola waveform.
    """
    start = params['start']
    width = params['width']
    height = params['height']

    y = np.zeros(width)

    for i in range(width):
        # Generate the parabola waveform within the specified range
        if start <= i < start + width:
            x = i - start  # Shift to start at x=0
            normalized_x = (x - width / 2) / (width / 2)  # Normalize to [-1, 1]
            y[i] = height * (1 - normalized_x**2)  # Parabolic equation: h(1 - (x/w)^2)
        else:
            y[i] = 0  # Zero outside the parabola range

    return y


if __name__ == "__main__":
    # Plot parabolas
    n = 101

    # Generate and plot the first parabola
    parabola1 = generate_parabola({'start': 10, 'width': 80, 'height': 15})
    plt.plot(parabola1, "g-", label="Parabola 1")

    # Generate and plot the second parabola
    parabola2 = generate_parabola({'start': 20, 'width': 50, 'height': 10})
    plt.plot(parabola2, "b--", label="Parabola 2")

    # Plot settings
    plt.xlabel('Number of Points')
    plt.ylabel('Amplitude')
    plt.title('Generated Parabolas')
    plt.legend()
    plt.show()
