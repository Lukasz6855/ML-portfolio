"""
Projekt: Przewidywanie cen domów (House Prices Regression)
Wykorzystanie PyCaret do automatycznego porównania modeli regresji
"""

import pandas as pd
from pycaret.regression import *

# Wczytanie danych
data = pd.read_csv('data/house_prices.csv')
print("Pierwsze wiersze danych:")
print(data.head())

# Inicjalizacja PyCaret (przygotowanie danych, podział train/test)
reg = setup(data=data, target='SalePrice', session_id=123)

# Porównanie wszystkich dostępnych modeli regresji
best_model = compare_models()

# Wyświetlenie wybranego modelu
print("\nNajlepszy model:")
print(best_model)

# Szczegółowa ocena modelu (MAE, RMSE, etc.)
evaluate_model(best_model)

# Zapisanie modelu do pliku (żeby móc go użyć później bez ponownego trenowania)
save_model(best_model, model_name='models/best_model')
print("Model został zapisany do pliku: models/best_model.pkl")

# Predykcje na danych testowych
predictions = predict_model(best_model)
print("\nPrzykładowe przewidywania cen:")
print(predictions[['SalePrice', 'prediction_label']].head(10))
