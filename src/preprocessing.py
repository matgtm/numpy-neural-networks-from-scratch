import numpy as np 
import pandas as pd
from pandas.api.types import is_numeric_dtype


    


def handle_missing_values(data,column_name, strategy, replace_with=None):
    '''Manejo de datos faltantes en un DataFrame.
    
    Parámetros:
    - data: pd.DataFrame -> DataFrame sobre el cual se aplicará la estrategia.
    - column_name: str -> Nombre de la columna a modificar.
    - strategy: str -> Estrategia a aplicar:
          - 'mean': reemplaza los NaN con el promedio (solo columnas numéricas)
          - 'median': reemplaza los NaN con la mediana (solo columnas numéricas)
          - 'most_frequent': reemplaza los NaN con el valor más frecuente
          - 'constant': reemplaza los NaN con el valor dado en `replace_with`
          - 'drop_col': elimina la columna del DataFrame
          - 'drop_rows': elimina las filas que tienen NaN en esa columna
    - replace_with: Valor con el cual reemplazar en la estrategia 'constant'.
    
    Retorna:
    - pd.DataFrame modificado según la estrategia aplicada.
    '''
    
    
    assert isinstance(data, pd.DataFrame), "Se espera que 'data' sea pd.DataFrame."
    
    col = data[column_name]
    
    if strategy == 'mean':
        if not is_numeric_dtype(col):
            return 'La columna debe ser numérica para calcular el promedio.'
        col = col.fillna(np.round(col.mean()))
        
    elif strategy == 'median':
        if not is_numeric_dtype(col):
            return 'La columna debe ser numérica para calcular el promedio.'
        col = col.fillna(np.round(col.median()))
        
    elif strategy == 'most_frequent':
        most_frequent = col.value_counts().idxmax()
        col = col.fillna(most_frequent)
        
    elif strategy == 'constant':
        if replace_with is None:
            raise ValueError('El valor con el que remplazar debe ser otorgado como parámetro.')
        col = col.fillna(replace_with)
    
    elif strategy == 'drop_col':
        data = data.drop(columns=column_name)
        return data
        
    elif strategy == 'drop_rows':
        data.dropna(subset=[column_name], inplace=True)
    
    data[column_name] = col
    return data

def ohe(data, column_name,values_list):
    for val in values_list:
        new_col_name = column_name + '_' + str(val)
        data[new_col_name] = (data[column_name] == val).astype(int)
    data = data.drop(columns=[column_name])
    return data

class Normalizer():
    '''
    '''
    
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, data):
        '''Guarda y calcula la media y desviación estándar de los datos.'''
        self.mean = data.mean()
        self.std = data.std()
        
    def transform(self, data):
        '''Aplica la normalización sobre el dataset con los parámetros ya guardados.'''
        return (data - self.mean) / self.std
    
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
    
    def inverse_transform(self, data_normalizada):
        data = data_normalizada * self.std + self.mean
        return data
    
    

