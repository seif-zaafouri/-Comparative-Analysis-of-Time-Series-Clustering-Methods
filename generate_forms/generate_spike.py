import matplotlib.pyplot as plt
import numpy as np


def generate_spike(params: dict):
    """
    Generates a spike waveform with a given duration and height.

    Args:
        params (dict): A dictionary containing the parameters for generating the spike waveform.
            - width (int): The duration of the spike waveform.
            - height (int): The height of the spike waveform.
            - center (int): The center index of the spike waveform in the time array.

    Returns:
        np.ndarray[float]: The generated spike waveform.

    """
    start = params['start']
    width = params['width']
    height = params['height']

    y = np.zeros(width)

    for i in range(width):
        # Generate the spike waveform from the start index of given width
        if i < start:
            y[i] = 0
        elif i < start + width//2:
            y[i] = (2*height*(i-start))/width
        elif i < start + width:
            y[i] = 2*height - (2*height*(i-start))/width
        else:
            y[i] = 0

    return y

if __name__ == "__main__":
    # Plot spike
    n = 101
    # Generate and plot the first spike
    spike1 = generate_spike({'width': 40, 'height': 10, 'start': 10})
    plt.plot(spike1, "rx", label="Spike 1")

    # Generate and plot the second spike
    spike2 = generate_spike({'width': 30, 'height': 10, 'start': 10})
    plt.plot(spike2, "bx", label="Spike 2")

    # Plot settings
    plt.xlabel('Nombre de points')
    plt.ylabel('Amplitude')
    plt.title('Generated Spikes')
    plt.legend()
    plt.show()
