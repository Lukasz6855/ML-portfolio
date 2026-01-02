"""
Projekt: Klasyfikacja przeżycia pasażerów Titanica
Wykorzystanie PyCaret do automatycznego porównania modeli klasyfikacji
"""

import pandas as pd
from pycaret.classification import *

# Wczytanie danych
data = pd.read_csv('data/titanic.csv')
print("Pierwsze wiersze danych:")
print(data.head())

# Inicjalizacja PyCaret (przygotowanie danych, podział train/test)
clf = setup(data=data, target='Survived', session_id=123)

# Porównanie wszystkich dostępnych modeli
best_model = compare_models()

# Wyświetlenie wybranego modelu
print("\nNajlepszy model:")
print(best_model)

# Szczegółowa ocena modelu
evaluate_model(best_model)

# Zapisanie modelu do pliku (żeby móc go użyć później bez ponownego trenowania)
save_model(best_model, model_name='models/best_model')
print("Model został zapisany do pliku: models/best_model.pkl")

# Predykcje na danych testowych
predictions = predict_model(best_model)
print("\nPrzykładowe przewidywania:")
print(predictions.head(10))
