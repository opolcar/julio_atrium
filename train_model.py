import pandas as pd
from models.NaiveModel import NaiveModel

df=pd.read_csv(
    filepath_or_buffer='C:\\Users\\artic\OneDrive\Escritorio\Ejercicio Julio Atrium\\mnist_784.csv',
    sep=';'
    )
objeto= NaiveModel()
objeto.fit(df)
objeto.save(filename='archivo_procesado.pkl')
print("El archivo se ha guardado correctamente.")