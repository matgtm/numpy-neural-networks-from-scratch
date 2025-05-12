import numpy as np
import pandas as pd


def train_val_split(dataset, val_fraction=0.2):
    """
    Separa un DataFrame en dos subconjuntos: uno para entrenamiento y otro para validación.

    Parámetros:
    -----------
    dataset : pd.DataFrame
        DataFrame que contiene los datos a dividir.
    val_fraction : float, opcional (default=0.2)
        Fracción del dataset que se asignará al conjunto de validación. 
        Por ejemplo, 0.2 indica que el 20% de los datos se usarán para validación y el 80% restante para entrenamiento.

    Retorna:
    --------
    tuple de pd.DataFrame
        (train, val) donde 'train' es el subconjunto de entrenamiento y 'val' el de validación.

    """
    
    assert isinstance(dataset, pd.DataFrame), "'dataset' debe ser un pd.DataFrame."
    
    train = dataset.sample(frac=1-val_fraction, random_state=42)
    val = dataset.drop(train.index)
    
    return train, val
