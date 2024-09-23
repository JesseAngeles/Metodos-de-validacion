import pandas as pd
from sklearn.model_selection import LeaveOneOut

# Separar el conjunto por Leave-One-Out
def split_by_LeaveOneOut(x_train, y_train):
    loo = LeaveOneOut()
    
    folds_train = []
    folds_val = []
    
    fold = 1
    for train_index, val_index in loo.split(x_train):
        X_train_fold = x_train.iloc[train_index]
        X_val_fold = x_train.iloc[val_index]
        y_train_fold = y_train.iloc[train_index]
        y_val_fold = y_train.iloc[val_index]
        
        df_train = pd.concat([X_train_fold.reset_index(drop=True), y_train_fold.reset_index(drop=True)], axis=1)
        df_val = pd.concat([X_val_fold.reset_index(drop=True), y_val_fold.reset_index(drop=True)], axis=1)
        
        # Agregar a las listas
        folds_train.append(df_train)
        folds_val.append(df_val)
        
        fold += 1

    return folds_train, folds_val
