import pandas as pd
from models.NaiveModel import NaiveModel

objeto = NaiveModel()
objeto.load(filename='archivo_procesado.pkl')
df=pd.read_csv(
    filepath_or_buffer='C:\\Users\\artic\OneDrive\Escritorio\Ejercicio Julio Atrium\\mnist_784.csv',
    sep=';'
    )
df_predict=objeto.predict(df=df)
df_predict.to_csv(path_or_buf='datos_inferidos.csv')
print('csv generado')