from models.naive_model import NaiveModel
import pandas as pd

# Una vez ejecutado naive_model.py ...

# Creamos el objeto
model = NaiveModel()

# Leemos el conjunto de datos
data = pd.read_csv('archives\CSV_original.csv')    # Este archivo no se sube a github porque ocupa demasiados mb y da error

# Ajustamos el modelo a los datos (calcular las medias)
model.fit(data)

# Guardamos el modelo (asumimos un archivo pickle (.pkl))
model.save('fit_model.pkl')

print('Proceso de entrenamiento finalizado, el modelo se ha guardado')

