
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_confusion_matrix(y_true, y_pred, plot=False):
    """
    Calcula la matriz de confusión y la visualiza.
    
    Parámetros:
    -----------
    y_true : pd.DataFrame o np.ndarray
        Etiquetas verdaderas, en formato one-hot o como valores discretos.
    y_pred : pd.DataFrame o np.ndarray
        Etiquetas predichas, en formato one-hot o como valores discretos.
    
    Retorna:
    --------
        Matriz de confusión visualizada como un heatmap.
    """
    # y_true
    if isinstance(y_true, pd.DataFrame):
        # Si tiene más de una columna, probablemente es one-hot
        if y_true.shape[1] > 1:
            y_true_labels = y_true.idxmax(axis=1)
        else:
            y_true_labels = y_true.iloc[:, 0]
    else:
        # Si es un array, comprobamos su dimensión
        if y_true.ndim > 1 and y_true.shape[1] > 1:
            y_true_labels = pd.Series(np.argmax(y_true, axis=1))
        else:
            y_true_labels = pd.Series(y_true)
    
    # y_pred
    if isinstance(y_pred, pd.DataFrame):
        if y_pred.shape[1] > 1:
            y_pred_labels = y_pred.idxmax(axis=1)
        else:
            y_pred_labels = y_pred.iloc[:, 0]
    else:
        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            y_pred_labels = pd.Series(np.argmax(y_pred, axis=1))
        else:
            y_pred_labels = pd.Series(y_pred)
            
    y_true_labels = y_true_labels.reset_index(drop=True)
    y_pred_labels = y_pred_labels.reset_index(drop=True)
    classes = np.unique(np.concatenate((y_true_labels, y_pred_labels)))
    # Inicializo la matriz de confusión
    cm = pd.DataFrame(
        data=np.zeros((len(classes), len(classes))),
        index=classes,
        columns=classes
    )
    
    # Sumo en cada celda
    for i in range(len(y_true_labels)):
        cm.loc[y_true_labels[i], y_pred_labels[i]] += 1
    
    cm = cm.astype(int)
    
    if plot:
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Matriz de Confusión')
        plt.xlabel('Predicción')
        plt.ylabel('Verdadero')
        plt.show()
    
    return cm

def accuracy(y_true, y_pred):
    """
    Calcula la accuracy del modelo, o sea, (TP + TN) / (TP + TN + FP + FN).
    
    Parámetros:
    -----------
    y_true : pd.DataFrame o np.ndarray
        Etiquetas verdaderas, en formato one-hot o como valores discretos.
    y_pred : pd.DataFrame o np.ndarray
        Etiquetas predichas, en formato one-hot o como valores discretos.
    
    Retorna:
    --------
        float: Precisión del modelo.
    """
    
    cm = get_confusion_matrix(y_true, y_pred)
    
    # Sumar los valores de la diagonal
    correct_predictions = cm.values.diagonal().sum()
    
    # Sumar todos los valores de la matriz
    total_predictions = cm.values.sum()
    
    accuracy = correct_predictions / total_predictions
    
    return np.round(accuracy,2)
