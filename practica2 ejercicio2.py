# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 14:56:39 2023

@author: luis mercado
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import sklearn.metrics as score
import pandas as pd
import numpy as np

# Funcion para calcular la especificidad
def specificity_score(y_test, preds):
    conf_matrix = score.confusion_matrix(y_test, preds)
    true_negatives = conf_matrix[0, 0]
    false_positives = conf_matrix[0, 1]
    return true_negatives / (true_negatives + false_positives)

def load_dataset(dataset_name):
    if dataset_name == "wine":
        dir_csv = 'vinos.csv'
        data = pd.read_csv(dir_csv, dtype=float)
        return data
    elif dataset_name == "diabetes":
        dir_csv = 'diabetes.csv'
        data = pd.read_csv(dir_csv, dtype=float, header=None)
        return data

    elif dataset_name == "seguro":
        dir_csv = 'autos.csv'
        data = pd.read_csv(dir_csv, dtype=float)
        return data
    else:
        return None

def perform_classification(data):
    if data is None:
        print("Dataset no válido.")
        return

    # Dividir los datos en X y Y
    X = np.array(data.iloc[:, :-1])  # Tomamos la última columna como nuestro target
    y = np.array(data.iloc[:, -1])

    # Dividir en sets de prueba y entrenamiento
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Estandarizar los parámetros para knn y svm
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Entrenar clasificadores
    logistic_regression = LogisticRegression(max_iter=1000)
    knn = KNeighborsClassifier(n_neighbors=10)
    svm = SVC(kernel='linear')
    naive_bayes = GaussianNB()

    logistic_regression.fit(X_train, y_train)
    knn.fit(X_train, y_train)
    svm.fit(X_train, y_train)
    naive_bayes.fit(X_train, y_train)

    # Predicciones
    logistic_regression_pred = logistic_regression.predict(X_test)
    knn_pred = knn.predict(X_test)
    svm_pred = svm.predict(X_test)
    naive_bayes_pred = naive_bayes.predict(X_test)

    print("Comprobacion: ")
    for i in range(min(len(X_test), 10)):
        print(f"Valor real: {y_test[i]}")
        print(f"\tPredicción Regresión Logística: {logistic_regression_pred[i]}")
        print(f"\tPredicción K Neighbors: {knn_pred[i]}")
        print(f"\tPredicción SVM: {svm_pred[i]}")
        print(f"\tPredicción Naive Bayes: {naive_bayes_pred[i]}")

def main():
    while True:
        print("\nSelecciona el dataset:")
        print("1. Calidad de Vinos")
        print("2. Diabetes")
        print("5. Seguros")
        print("6. Salir")

        choice = input("Ingresa el número correspondiente al dataset o '3' para salir: ")

        if choice == '1':
            print("\n>>>>>vinos<<<<<")
            data = load_dataset("wine")
            perform_classification(data)
        elif choice == '2':
            print("\n>>>>>diabetes<<<<<<")
            data = load_dataset("diabetes")
            perform_classification(data)
        elif choice == '5':
            print("\n>>>>>>seguros<<<<<<<<")
            data = load_dataset("seguro")
            print("no se puede clasificar")
        elif choice == '6':
            print("Saliendo...")
            break
        else:
            print("Opción inválida. Por favor, elige una opción válida.")

if __name__ == "__main__":
    main()
