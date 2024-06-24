import pandas as pd
from pandas import DataFrame
import pickle

class NaiveModel:

    def __init__(self):
        self.media={}

    def fit(self, df:DataFrame):
        '''Cálculo media columnas'''
        self.media = df.mean().to_dict()

    def predict(self, df: DataFrame) -> DataFrame:
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

    def save(self, filename:str): 
        with open (filename, 'wb') as file:
            pickle.dump(self.media, file)
        print(f'Medias guardadas en {filename}')
        
    def load(self,filename:str):
        with open(filename, 'rb') as f:
            self.media = pickle.load(f)
    
    