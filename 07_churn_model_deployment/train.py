"""
🎯 Trening modelu Churn - wersja produkcyjna

Ten skrypt trenuje model churn prediction i zapisuje go do użycia w produkcji.
"""

import pandas as pd
from pycaret.classification import *
import json

print("="*80)
print("🎓 TRENING MODELU CHURN")
print("="*80)

# ============================================================================
# WCZYTANIE I PRZYGOTOWANIE DANYCH
# ============================================================================

print("\n📂 Wczytywanie danych...")

# Wczytanie datasetu
df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

print(f"✅ Wczytano {len(df)} klientów, {len(df.columns)} kolumn")

# Czyszczenie danych
df = df.drop('customerID', axis=1)  # Usunięcie ID
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')  # Konwersja na liczby
df['TotalCharges'].fillna(0, inplace=True)  # Wypełnienie braków

print(f"✅ Dane przygotowane")

# ============================================================================
# KONFIGURACJA PYCARET
# ============================================================================

print("\n⚙️ Konfiguracja PyCaret...")

clf_setup = setup(
    data=df,
    target='Churn',
    session_id=123,
    train_size=0.8,
    fold=5,
    normalize=True,
    verbose=False
)

print("✅ PyCaret skonfigurowany (80/20 train/test split)")

# ============================================================================
# WYBÓR I TRENING MODELU
# ============================================================================

print("\n🔄 Porównywanie modeli (optymalizacja pod Recall)...")

# Wybór najlepszego modelu pod kątem Recall
best_model = compare_models(sort='Recall', n_select=1)

print(f"\n✅ Najlepszy model: {type(best_model).__name__}")

# Trening finalnego modelu
print("\n🎓 Trening finalnego modelu...")
final_model = create_model(best_model)

print("✅ Model wytrenowany!")

# ============================================================================
# ZAPIS MODELU
# ============================================================================

print("\n💾 Zapisywanie modelu...")

model_filename = 'churn_model'
save_model(final_model, f'models/{model_filename}')

print(f"✅ Model zapisany: models/{model_filename}.pkl")

# ============================================================================
# ZAPIS METADANYCH
# ============================================================================

print("\n💾 Zapisywanie metadanych...")

metadata = {
    "threshold": 0.5,
    "optimized_for": "recall",
    "business_reason": "false negatives are costly",
    "model_type": type(final_model).__name__,
    "train_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
}

with open('models/metadata.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print("✅ Metadata zapisane: models/metadata.json")

# ============================================================================
# PODSUMOWANIE
# ============================================================================

print("\n" + "="*80)
print("🎉 TRENING ZAKOŃCZONY POMYŚLNIE!")
print("="*80)
print(f"\n📦 Zapisane pliki:")
print(f"   - models/{model_filename}.pkl (wytrenowany model)")
print(f"   - models/metadata.json (ustawienia)")
print(f"\n🚀 Model gotowy do użycia w predict.py!")
print("="*80)
