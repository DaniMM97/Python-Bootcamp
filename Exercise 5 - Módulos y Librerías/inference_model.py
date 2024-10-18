from models.naive_model import NaiveModel
import pandas as pd


# Tras ejercutar train_model.py ...

# Creamos el modelo
model2 = NaiveModel()

# Cargamos el modelo que acabamos de entrenar
model2.load('fit_model.pkl')

# Leemos los datos para llevar a cabo la inferencia en cuestión
data = pd.read_csv('archives\CSV_original.csv')

# Realizamos la inferencia
predictions = model2.predict(data)

# Guardamos las predicciones en un archivo CSV
predictions.to_csv('predictions.csv', index=False)  # Guardar las predicciones sin índices

print('Proceso de inferencia finalizado, se ha generado un nuevo archivo')
