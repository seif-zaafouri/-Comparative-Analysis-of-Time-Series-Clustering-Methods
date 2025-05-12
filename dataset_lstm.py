import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from generate_forms.generate_spike import generate_spike
from generate_forms.generate_M import generate_M
from generate_forms.generate_CLDR_CRDL import generate_CLDR, generate_CRDL
from generate_forms.generate_parabola import generate_parabola

class DatasetLSTM:
    def __init__(
        self, 
        formsPerClass = 5,
        noise_levels  = [0.0, 0.02, 0.1], 
        widths_list = [100, 200, 300, 500],
        width_variation = [50, 50, 50, 50],
    ):
        self.formsPerClass = formsPerClass
        self.noise_levels = noise_levels
        self.widths_list = widths_list
        self.width_variation = width_variation

    # Save the properties of the DatasetLSTM instance to a file
    def save_dataset(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.__dict__, file)
    
    # Load the properties of the DatasetLSTM instance from a file
    @staticmethod
    def load_dataset(filename):
        with open(filename, 'rb') as file:
            data = pickle.load(file)
        instance = DatasetLSTM()
        instance.__dict__.update(data)
        return instance
    
    # Create a DataFrame with randomly generated forms
    def createDataFrame(self, start = False):
        if start :
            s = 0.2*self.widths_list[0]
        else :
            s = 0
        base_params_m = {'width': None, 'height': 50, 'start': 0}
        variation_range_m = {'width': None, 'height': 20, 'start': s}
        
        base_spike_params = {'width': None, 'height': 50, 'start': 0}
        variation_spike_range = {'width': None, 'height': 20, 'start': s}
        
        base_parabola_params = {'width': None, 'height': 50, 'start': 0}
        variation_parabola_range = {'width': None, 'height': 20, 'start': s}

        base_cldr_params = {'width': None, 'height': 50, 'start': 0, 'middle': 0.8, 'A': 0.03}
        variation_cldr_range = {'width': None, 'height': 20, 'start': s, 'middle': 0.15, 'A': 0.02}

        base_crdl_params = {'width': None, 'height': 50, 'start': 0, 'middle': 0.8, 'A': 0.03}
        variation_crdl_range = {'width': None, 'height': 20, 'start': s, 'middle': 0.15, 'A': 0.02}
        
        params = {
            "M": (generate_M, base_params_m, variation_range_m),
            "Spike": (generate_spike, base_spike_params, variation_spike_range),
            "Parabola": (generate_parabola, base_parabola_params, variation_parabola_range),
            "CLDR": (generate_CLDR, base_cldr_params, variation_cldr_range),
            "CRDL": (generate_CRDL, base_crdl_params, variation_crdl_range)
        }

        data = []
        # Iterate over the list of points
        for i, width in enumerate(self.widths_list):
            # Change widths in the base parameters
            for form_name, (generate_form, base_params, variation_range) in params.items():
                base_params['width'] = width
                variation_range['width'] = self.width_variation[i]
                
            # Iterate over the list of noise levels
            for noise_level in self.noise_levels:
                # Iterate over each form type and its corresponding parameters
                for form_name, (generate_form, base_params, variation_range) in params.items():
                    # Generate forms with added noise
                    forms = [
                    self.add_noise(
                        generate_form(self.generate_random_params(base_params, variation_range)), 
                        noise_level)
                    for _ in range(self.formsPerClass)
                    ]
                    # Extend the data list with the generated forms and their metadata
                    data.extend([{'Form': f, 'Class': form_name, 'Noise_Level': noise_level, 'N_Points': len(f)} for f in forms])

        # Create a DataFrame from the data list
        self.df = pd.DataFrame(data)
        return self

    # Generate random parameters with constraints
    def generate_random_params(self, base_params, variation_range):
        random_params = {}
        for key in base_params:
            base_value = base_params[key]
            variation = variation_range[key]
            
            if isinstance(base_value, int):
                random_params[key] = np.random.randint(base_value - variation, base_value + variation+1)
            else:
                random_params[key] = np.random.uniform(base_value - variation, base_value + variation)
        return random_params
    

    # Add noise to a signal
    def add_noise(self, signal, noise_level):
        noise = np.random.normal(0, noise_level, len(signal))
        # return signal + noise * max(signal)
        return signal + noise

    def pad_data(self, target_length=None, mask_value=0):
        if target_length is None:
            target_length = self.df['N_Points'].max()
        # Ajouter une colonne avec les courbes complétées
        self.df['Padded_Form'] = [
            np.pad(curve, (0, target_length - n_points), constant_values=mask_value)
            for curve, n_points in zip(self.df['Form'], self.df['N_Points'])
        ]

        return self
    
    # Get the X matrix and the one-hot encoded y vector
    def get_X_y(self, split=True, train_size=0.7, test_size=0.2):
        try:
            X = np.array(self.df['Padded_Form'].to_list())

            # Reshape the data to fit the LSTM input shape
            X = X.reshape(X.shape[0], X.shape[1], 1)
            
            self.encoder = OneHotEncoder(sparse_output=False)
            y = self.encoder.fit_transform(self.df[['Class']])
            
            if split:
                return self.split_dataset(X, y, train_size, test_size)
            else:
                return X, y
        except Exception as e:
            print("Please run the pad_data method first")
            print(e)


    # Split the dataset into training, testing and validation sets
    def split_dataset(self, X, y, train_size=0.6, test_size=0.15):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        # train_size = param*(1-test_size)
        # Donc on passe en param : param = train_size/(1-test_size)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=train_size/(1-test_size))

        return X_train, y_train, X_test, y_test, X_val, y_val
    

    # Decode the one-hot encoded labels
    def decode_labels(self, y):
        return self.encoder.inverse_transform(y)
    
    
    def plot_length_repartition(self):
        # Check that all forms have the same length
        # print(self.df.groupby(lambda l: len(self.df.loc[l, 'Padded_Form'])).size())
        
        # Plot the histogram of the lengths
        self.df['N_Points'].plot.hist(
            bins=20,
            title="Répartition initiale",
            figsize=(5, 3),
            xlabel="Longueur des formes",
        )

    # Normalize all data in the DataFrame
    def normalizeData(self):
        # Convert the forms to a numpy array
        data = np.array(self.df['Form'].values.tolist())
        
        # Fit the scaler to the data and transform it
        self.scaler = StandardScaler()
        data_normalized = self.scaler.fit_transform(data)
        # Add the normalized data to the DataFrame
        self.df['Form_normalized'] = data_normalized.tolist()

        return self
    
    def copy(self):
        dataset_copy = DatasetLSTM(
            formsPerClass=self.formsPerClass,
            noise_levels=self.noise_levels,
            widths_list=self.widths_list,
            width_variation=self.width_variation
        )
        dataset_copy.df = self.df.copy()
        return dataset_copy
    

if __name__=="__main__":
    dataset_wn = DatasetLSTM(
        formsPerClass=2,
        noise_levels  = [0.0], 
        widths_list = [300],
        width_variation=[200]
    ).createDataFrame().pad_data()

    print("Nombre de formes :", len(dataset_wn.df))