from pandas import DataFrame
import pickle

class NaiveModel:

    def __init__(self):
        self.media={}
        self.archivo_procesado='archivo_procesado.pkl'

    def fit(self, df:DataFrame):
        '''Cálculo media columnas'''
        self.media = df.mean().to_dict()

    def predict(self, df: DataFrame) -> DataFrame:
        '''División de cada dato por la media de su columna'''
        df_copy = df.copy(deep=True).astype(float)
        columns=df_copy.columns
        rows=len(df_copy)
        for column in columns:
            media_columna=self.media.get(column)
            if media_columna == 0:
                continue
            for row in range(0,rows,):
                if df_copy[column][row] == 0:
                    continue
                df_copy.loc[row, column]=df_copy[column][row]/media_columna
        return df_copy

    def save(self): 
        '''Guardar pickle con las medias de predict'''
        with open (self.archivo_procesado, 'wb') as file:
            pickle.dump(self.media, file)
        print(f'Medias guardadas en {self.archivo_procesado}')
        
    def load(self):
        '''Cargar datos del archivo pickle'''
        with open(self.archivo_procesado, 'rb') as f:
            self.media = pickle.load(f)
    
    