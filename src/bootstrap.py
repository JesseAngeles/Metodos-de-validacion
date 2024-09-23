import pandas as pd
from sklearn.utils import resample

# Separar el conjunto por bootstrap
def split_by_bootstrap(x_train, y_train, n_samples=5, n_sets=2):
    folds_train = []
    folds_val = []
    
    # Generar muestras bootstrap
    for _ in range(n_sets): 
        # Obtener un conjunto bootstrap
        bootstrap_samples = resample(x_train, n_samples=n_samples, replace=True)
        
        bootstrap_indices = bootstrap_samples.index
        X_bootstrap = x_train.iloc[bootstrap_indices]
        y_bootstrap = y_train.iloc[bootstrap_indices]

        # Obtener los índices de los registros que no están en el conjunto de bootstrap
        remaining_indices = x_train.index.difference(bootstrap_indices)

        # Obtener el conjunto de validación de los registros restantes
        X_val = x_train.loc[remaining_indices]
        y_val = y_train.loc[X_val.index]  # Asegúrate de que las etiquetas correspondan

        df_bootstrap = pd.concat([X_bootstrap.reset_index(drop=True), y_bootstrap.reset_index(drop=True)], axis=1)
        df_val = pd.concat([X_val.reset_index(drop=True), y_val.reset_index(drop=True)], axis=1)

        # Agregar a las listas
        folds_train.append(df_bootstrap)
        folds_val.append(df_val)  # Agregar el conjunto de validación completo

    return folds_train, folds_val