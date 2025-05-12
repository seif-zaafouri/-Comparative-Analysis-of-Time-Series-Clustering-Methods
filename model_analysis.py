import numpy as np
import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score, precision_score, recall_score

from dataset_lstm import DatasetLSTM

class ModelAnalysis:
    def __init__(self, dataset:DatasetLSTM):
        self.dataset = dataset

    # Save the model to a file
    def save_model(self, path):
        self.model.save(path)
        return self
    
    # Load the model from a file
    @staticmethod
    def load_model(model_path, dataset_path):
        model = keras.models.load_model(model_path)
        dataset = DatasetLSTM.load_dataset(dataset_path)
        instance = ModelAnalysis(dataset)
        instance.model = model
        return instance
    
    def prepare_data(self, train_size=0.7, test_size=0.15, print_info=False):
        X_train, y_train, X_test, y_test, X_val, y_val = self.dataset.get_X_y(train_size=train_size, test_size=test_size)
        
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test
        self.X_val, self.y_val = X_val, y_val

        if print_info:
            m = X_train.shape[0] + X_test.shape[0] + X_val.shape[0]
            print(f"Dimensions de X_train, y_train ({100*X_train.shape[0]/m:.2f}%) :", X_train.shape, y_train.shape)
            print(f"Dimensions de X_test, y_test ({100*X_test.shape[0]/m:.2f}%) :", X_test.shape, y_test.shape)
            print(f"Dimensions de X_val, y_val ({100*X_val.shape[0]/m:.2f}%) :", X_val.shape, y_val.shape)
        
        return self

    def create_model(self, layers_list, name="LSTM_Model", summary=True):
        model = keras.Sequential(name=name)
        # Input layer
        model.add(layers.Input(shape=(self.X_train.shape[1], self.X_train.shape[2]), name='Input_Layer'))
        
        # Add the layers
        for layer in layers_list:
            model.add(layer)

        # Output layer
        model.add(layers.Dense(self.y_train.shape[1], activation='softmax', name='Output_Layer'))

        if summary:
            model.summary()
        
        self.model = model
        return self
    
    def train_model(
            self, optimizer='adam', 
            loss='categorical_crossentropy', metrics=['accuracy'], 
            epochs=20, batch_size=32
        ):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        history = self.model.fit(self.X_train, self.y_train, 
                                 epochs=epochs, batch_size=batch_size, 
                                 validation_data=(self.X_val, self.y_val), shuffle=True)

        self.history = history
        return self

    def plot_summary(self):
        self.model.summary()

 
    # Evaluate the model on the test set
    def evaluate_model(self):
        _, test_accuracy = self.model.evaluate(self.X_test, self.y_test)
        print(f'Test Accuracy: {(test_accuracy*100):.2f}%')
        return self
    
    def evaluate(self, set_name):
        if set_name == 'train':
            X, y = self.X_train, self.y_train
        elif set_name == 'test':
            X, y = self.X_test, self.y_test
        elif set_name == 'val':
            X, y = self.X_val, self.y_val
        else:
            print("Invalid set name")
            return
        
        _, accuracy = self.model.evaluate(X, y)
        print(f'{set_name.capitalize()} Accuracy: {(accuracy*100):.2f}%')
        return self

    # Plot the model learning history
    def plot_history(self):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss Evolution')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy Evolution')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self):
        y_pred = self.model.predict(self.X_test)
        y_pred_decoded = self.dataset.decode_labels(y_pred)
        y_test_decoded = self.dataset.decode_labels(self.y_test)
        matrix = confusion_matrix(y_test_decoded, y_pred_decoded)
        ConfusionMatrixDisplay(matrix, display_labels=self.dataset.encoder.categories_[0]).plot()
        plt.xlabel('Predicted Classes')
        plt.ylabel('True Classes')
        plt.title(f'Confusion Matrix (Accuracy: {100*np.trace(matrix) / np.sum(matrix):.2f}%)')
        plt.show()

    def plot_scores(self):
        # plot f1 score, precision, recall for each class
        y_pred = self.model.predict(self.X_test)
        y_pred_decoded = self.dataset.decode_labels(y_pred)
        y_test_decoded = self.dataset.decode_labels(self.y_test)
        f1 = f1_score(y_test_decoded, y_pred_decoded, average=None)
        precision = precision_score(y_test_decoded, y_pred_decoded, average=None)
        recall = recall_score(y_test_decoded, y_pred_decoded, average=None)
        classes = self.dataset.encoder.categories_[0]
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.bar(classes, f1)
        plt.title('F1 Score')
        plt.subplot(1, 3, 2)
        plt.bar(classes, precision)
        plt.title('Precision')
        plt.subplot(1, 3, 3)
        plt.bar(classes, recall)
        plt.title('Recall')
        plt.tight_layout()
        plt.show()

    def predict_random_curves(self, n = 5):
        class_colors = {
            'M': 'blue',
            'Spike': 'green',
            'CLDR': 'red',
            'CRDL': 'cyan',
            'Parabola': 'magenta'
        }
        
        # Select n random curves from the test set
        indices = np.random.choice(range(self.X_test.shape[0]), n, replace=False)
        curves = self.X_test[indices, :, :]

        classes = self.dataset.decode_labels(self.y_test[indices]).T[0]

        for i in range(n):
            plt.plot(curves[i, :, 0], label=classes[i], color=class_colors[classes[i]])
        plt.legend()
        plt.title("Random curves prediction with LSTM")
        plt.show()

    # récupérer et afficher les courbes de test mal predites
    def plotErrors(self):
        class_colors = {
            'M': 'blue',
            'Spike': 'green',
            'CLDR': 'red',
            'CRDL': 'cyan',
            'Parabola': 'magenta'
        }
         
        y_pred = self.model.predict(self.X_test)
        y_pred_decoded = self.dataset.decode_labels(y_pred)
        y_test_decoded = self.dataset.decode_labels(self.y_test)

        errors = []
        for i in range(len(y_pred_decoded)):
            if y_pred_decoded[i] != y_test_decoded[i]:
                errors.append(i)
        print(f"Nombre de courbes mal prédites : {len(errors)}")
        print(f"Pourcentage de courbes mal prédites : {100*len(errors)/len(y_pred_decoded):.2f}%")

        # Afficher les courbes mal predites
        for i in errors:
            plt.plot(self.X_test[i, :, 0], 
                     label=f"True: {y_test_decoded[i][0]} - Pred: {y_pred_decoded[i][0]}", 
                     color=class_colors[y_test_decoded[i][0]])
        plt.legend()
