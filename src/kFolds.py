import pandas as pd
from sklearn.model_selection import KFold

# Separar el conjunto por KFold
def split_by_KFold(x_train, y_train, n_splits = 3):
    kf = KFold(n_splits)
    
    folds_train = []
    folds_val = []
    
    for fold, (train_index, val_index) in enumerate(kf.split(x_train), start=1):
        X_train_fold = x_train.iloc[train_index]
        X_val_fold = x_train.iloc[val_index]
        y_train_fold = y_train.iloc[train_index]
        y_val_fold = y_train.iloc[val_index]
        
        # Unir características y etiquetas para el conjunto de entrenamiento y validación
        df_train = pd.concat([X_train_fold.reset_index(drop=True), y_train_fold.reset_index(drop=True)], axis=1)
        df_val = pd.concat([X_val_fold.reset_index(drop=True), y_val_fold.reset_index(drop=True)], axis=1)
        
        # Agregar los DataFrames a las listas
        folds_train.append(df_train)
        folds_val.append(df_val)
        
    return folds_train, folds_val