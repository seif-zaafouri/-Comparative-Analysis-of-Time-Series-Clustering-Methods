# Créer un DataFrame avec des formes générées aléatoirement
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

from generate_forms.generate_spike import generate_spike
from generate_forms.generate_M import generate_M
from generate_forms.generate_CLDR_CRDL import generate_CLDR, generate_CRDL
from generate_forms.generate_parabola import generate_parabola


# Generate random parameters with constraints
def generate_random_params(base_params, variation_range):
    random_params = {}
    for key in base_params:
        base_value = base_params[key]
        variation = variation_range[key]
        random_value = np.clip(
            np.random.uniform(base_value - variation, base_value + variation),
            1,  # Avoid zero or negative values for parameters like width
            np.inf,
        )
        random_params[key] = int(random_value) if key in ['start', 'width'] else random_value
    return random_params


# Add noise to a signal
def add_noise(signal, noise_level):
    noise = np.random.normal(0, noise_level, len(signal))
    return signal + noise * max(signal)


# Standardize signal lengths
def standardize_signal_length(signals, target_length):
    standardized_signals = []
    for signal in signals:
        if len(signal) < target_length:
            signal = np.pad(signal, (0, target_length - len(signal)), mode='constant')
        elif len(signal) > target_length:
            signal = signal[:target_length]
        standardized_signals.append(signal)
    return standardized_signals


# Base parameters and variation ranges for each form
base_params_m = {'start': 50, 'width': 100, 'height': 50}
variation_range_m = {'start': 5, 'width': 50, 'height': 20}

base_spike_params = {'width': 100, 'height': 30, 'start': 50}
variation_spike_range = {'width': 50, 'height': 10, 'start': 5}

base_cl_params = {'start': 20, 'width': 100, 'height': 40, 'A': 0.2, 'ratio': 0.6}
variation_cl_range = {'start': 5, 'width': 50, 'height': 10, 'A': 0.2, 'ratio': 0.3}

base_parabola_params = {'start': 50, 'width': 100, 'height': 50}
variation_parabola_range = {'start': 5, 'width': 50, 'height': 20}


# Créer un DataFrame avec des formes générées aléatoirement
# formsPerClass est le nombre de formes générées pour chaque classe (par niveau de bruit et par nombre de points)
# Si formsPerClass=5, alors il y aura 5 formes générées pour chaque classe (M, Spike, CLDR, CRDL, Parabola) 
# Donc 25 formes générées pour chaque niveau de bruit et chaque nombre de points
def createDataFrame(noise_levels, n_points_list, formsPerClass=5):
    data = []
    for n_points in n_points_list:
        for noise_level in noise_levels:
            # Generate forms for each class
            m_forms = [
                add_noise(generate_M(n_points, generate_random_params(base_params_m, variation_range_m)), noise_level)
                for _ in range(formsPerClass)
            ]
            spike_forms = [
                add_noise(
                    generate_spike(n_points, generate_random_params(base_spike_params, variation_spike_range)),
                    noise_level,
                )
                for _ in range(formsPerClass)
            ]
            # cldr_forms = [
            #     add_noise(
            #         generate_CLDR(n_points, generate_random_params(base_cl_params, variation_cl_range)),
            #         noise_level,
            #     )
            #     for _ in range(formsPerClass)
            # ]
            # crdl_forms = [
            #     add_noise(
            #         generate_CRDL(n_points, generate_random_params(base_cl_params, variation_cl_range)),
            #         noise_level,
            #     )
            #     for _ in range(formsPerClass)
            # ]
            parabola_forms = [
                add_noise(
                    generate_parabola(n_points, generate_random_params(base_parabola_params, variation_parabola_range)),
                    noise_level,
                )
                for _ in range(formsPerClass)
            ]

            # Collect data
            for form, label in zip(
                # [m_forms, spike_forms, cldr_forms, crdl_forms, parabola_forms],
                # ['M', 'Spike', 'CLDR', 'CRDL', 'Parabola'],
                [m_forms, spike_forms, parabola_forms],
                ['M', 'Spike', 'Parabola'],
            ):
                data.extend([{'Form': f, 'Class': label, 'Noise_Level': noise_level, 'N_Points': n_points} for f in form])

    # Create DataFrame
    df = pd.DataFrame(data)
    return df







# Normaliser toutes les données sauf les paraboles
def normalizeData(df):
	# Sélectionner les données qui ne sont pas des paraboles
    mask_not_parabola = df['Class'] != 'Parabola'
    data_to_normalize = np.array(df.loc[mask_not_parabola, 'Form'].values.tolist())
    
    # Normaliser les données sélectionnées
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data_to_normalize.T).T  # Transpose pour normaliser correctement
    
    # Remettre les données normalisées dans le DataFrame
    df.loc[mask_not_parabola, 'Form'] = data_normalized.tolist()
    
    return df, scaler

# Normaliser toutes les données
def normalizeAllData(df, target_length=None):
    # Determine the target length (use the maximum length if not provided)
    if target_length is None:
        target_length = max(len(signal) for signal in df['Form'])

    # Standardize the lengths of all signals
    standardized_forms = []
    for signal in df['Form']:
        if len(signal) < target_length:
            # Pad with zeros if the signal is too short
            signal = np.pad(signal, (0, target_length - len(signal)), mode='constant')
        elif len(signal) > target_length:
            # Truncate if the signal is too long
            signal = signal[:target_length]
        standardized_forms.append(signal)

    # Convert to NumPy array
    data = np.array(standardized_forms)

    # Normalize the data
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data)

    # Update the DataFrame
    df['Form'] = list(data_normalized)

    return df, scaler

def plot_signals_separately(df, samples_per_class=3):
    """
    Plots signals, each in a separate figure.

    Args:
        df (pd.DataFrame): DataFrame containing the signals.
        samples_per_class (int): Number of samples to plot per class.
    """
    classes = df['Class'].unique()

    for form_class in classes:
        class_df = df[df['Class'] == form_class]

        # Select samples for the class
        selected_samples = class_df.sample(n=min(samples_per_class, len(class_df)))

        for idx, (_, row) in enumerate(selected_samples.iterrows()):
            plt.figure(figsize=(10, 6))
            plt.plot(row['Form'], label=f"Class: {row['Class']} | Noise: {row['Noise_Level']} | Points: {row['N_Points']}")
            plt.title(f"Sample {idx + 1} - Class: {row['Class']}")
            plt.xlabel("Points")
            plt.ylabel("Amplitude")
            plt.legend()
            plt.grid(True)
            plt.show()

            
def plot_signals_together(df, samples_per_class=3):
    """
    Plots signals of each class in a single figure with subplots for different items.

    Args:
        df (pd.DataFrame): DataFrame containing the signals.
        samples_per_class (int): Number of samples to plot per class.
    """
    classes = df['Class'].unique()

    for form_class in classes:
        class_df = df[df['Class'] == form_class]
        fig, axes = plt.subplots(1, samples_per_class, figsize=(15, 5))

        # Select samples for the class
        selected_samples = class_df.sample(n=min(samples_per_class, len(class_df)))

        for j, (_, row) in enumerate(selected_samples.iterrows()):
            ax = axes[j]
            ax.plot(row['Form'], label=f"Noise: {row['Noise_Level']} | Points: {row['N_Points']}")
            ax.set_title(f"Class: {row['Class']} - Sample {j + 1}")
            ax.set_xlabel("Points")
            ax.set_ylabel("Amplitude")
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        plt.show()




if __name__ == "__main__":
    # Parameters
    noise_levels = [0.0, 0.02, 0.1]
    n_points_list = [100, 150, 200] 
    df = createDataFrame(noise_levels=noise_levels, n_points_list=n_points_list)
    # plot_signals_separately(df, samples_per_class=5)
    plot_signals_together(df, samples_per_class=3)
       
