import pandas as pd
from sklearn.model_selection import train_test_split

from kFolds import split_by_KFold
from leaveOneOut import split_by_LeaveOneOut
from bootstrap import split_by_bootstrap

def printMenu():
    print("0) Exit")
    print("1) Split by K-Folds")
    print("2) Split by Leave-One-Out")
    print("3) Split by Bootstrap")
    return input()

def printValidationGroups(folds_train, folds_val):
    for i in range(len(folds_train)):
        print(f"\nConjunto de entrenamiento {i+1}:\n", folds_train[i])
        print(f"Conjunto de validación {i+1}:\n", folds_val[i])

# Leer el archivo CSV
df = pd.read_csv('./src/resources/metodosDeValidacion.csv')

# Se extraen los encabezados
headers = df.columns.to_list()

# Se extraen las características
X = df[headers[0:-1]]
# Se extraen las etiquetas
y = df[headers[-1]]

# Dividir en conjuntos de entrenamiento y pruebas
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle=False)

while True:
    option = printMenu()
    if option == "0":
        print("bye")
        break
    elif option == "1":
        folds_train, folds_val = split_by_KFold(x_train, y_train)
    elif option == "2":
        folds_train, folds_val = split_by_LeaveOneOut(x_train, y_train)
    elif option == "3":
        folds_train, folds_val = split_by_bootstrap(x_train, y_train)
    else:
        continue
     
    printValidationGroups(folds_train, folds_val)
