import pickle
import pandas as pd


class NaiveModel:
    def __init__(self):
        """Inicializa la clase (el atributo 'means' (medias))"""
        self.means = None

    def fit(self, data):   # data = dataframe (CSV)
        """Calcula la media de cada columna del dataframe y la almacena en el atributo 'means' (en el self)"""
        self.means = data.mean()

    def predict(self, modified_data):
        """Divide cada elemento de la columna por su media correspondiente. Devuelve una copia"""
        modified_data_copy = modified_data.copy()   # Copia profunda en una variable
        
        for element in modified_data_copy.columns:
            modified_data_copy[element] = modified_data_copy[element] / self.means[element]

        return modified_data_copy

    def save(self, new_document):
        """Guarda las medias calculadas utilizando 'pickle'"""
        with open(new_document, 'wb') as f:
            pickle.dump(self.means, f)

    def load(self, new_document):
        """Carga las medias guardadas desde el propio del ordenador, para hacer el 'predict' sin volver a usar 'fit'"""
        with open(new_document, 'rb') as f:
            self.means = pickle.load(f)