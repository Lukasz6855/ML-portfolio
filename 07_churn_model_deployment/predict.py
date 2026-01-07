"""
🔮 Predykcje Churn dla nowych klientów

Ten skrypt wczytuje zapisany model i wykonuje predykcje dla nowych klientów.
WAŻNE: prediction_score = prawdopodobieństwo dla predicted class (nie zawsze dla "Yes")!
"""

import pandas as pd
from pycaret.classification import *
import json

print("="*80)
print("🔮 PREDYKCJE CHURN")
print("="*80)

# ============================================================================
# PRZYGOTOWANIE NOWYCH KLIENTÓW
# ============================================================================

print("\n📂 Przygotowywanie nowych klientów...")

# Wczytanie oryginalnego datasetu
df_original = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Losowanie 20 klientów (więcej = bardziej zróżnicowana próbka)
new_customers = df_original.sample(20, random_state=123)

# Usunięcie kolumny Churn (symulacja danych produkcyjnych)
new_customers = new_customers.drop('Churn', axis=1)

# Zapis do pliku
new_customers.to_csv('data/new_customers.csv', index=False)

print(f"✅ Przygotowano {len(new_customers)} nowych klientów")
print("💾 Zapisano: data/new_customers.csv")

# ============================================================================
# WCZYTANIE MODELU I METADANYCH
# ============================================================================

print("\n📂 Wczytywanie modelu...")

# Wczytanie modelu
model = load_model('models/churn_model')

print(f"✅ Model wczytany: {type(model).__name__}")

# Wczytanie metadanych
with open('models/metadata.json', 'r', encoding='utf-8') as f:
    metadata = json.load(f)

threshold = metadata['threshold']

print(f"✅ Metadata wczytane (threshold: {threshold})")

# ============================================================================
# WCZYTANIE NOWYCH KLIENTÓW
# ============================================================================

print("\n📂 Wczytywanie klientów do oceny...")

customers = pd.read_csv('data/new_customers.csv')

print(f"✅ Wczytano {len(customers)} klientów")

# ============================================================================
# PREDYKCJE
# ============================================================================

print("\n🔮 Wykonywanie predykcji...")
print(f"🎯 Używam threshold z metadata.json: {threshold}")

# WAŻNE: Przekazujemy threshold z metadata.json!
# Bez tego PyCaret użyłby domyślnego threshold = 0.5
predictions = predict_model(model, data=customers, probability_threshold=threshold)

print("✅ Predykcje zakończone")

# ============================================================================
# ANALIZA WYNIKÓW
# ============================================================================

print(f"\n📊 Analiza wyników z threshold: {threshold}")

# Wybór kluczowych kolumn
result_columns = ['customerID', 'tenure', 'MonthlyCharges', 'Contract', 
                  'prediction_score', 'prediction_label']

results = predictions[result_columns].copy()

# Zaokrąglenie prawdopodobieństwa
results['prediction_score'] = results['prediction_score'].round(4)

# WAŻNE: prediction_score = prawdopodobieństwo dla predicted class!
# - Jeśli prediction_label = Yes → score = prawdopodobieństwo ODEJŚCIA
# - Jeśli prediction_label = No → score = prawdopodobieństwo POZOSTANIA

# Dodanie wyjaśnień
results['explanation'] = results.apply(
    lambda row: f"Przewiduje: {row['prediction_label']} (pewność: {row['prediction_score']:.2%})",
    axis=1
)

# Dodanie poziomów ryzyka - TYLKO dla prediction_label = Yes
def get_risk_level(prob):
    if prob >= 0.7:
        return "HIGH"
    elif prob >= 0.5:
        return "MEDIUM"
    else:
        return "LOW"

def get_action(prob):
    if prob >= 0.7:
        return "PILNE: Kontakt retencji + oferta specjalna"
    elif prob >= 0.5:
        return "Kontakt telefoniczny + analiza"
    else:
        return "Monitoring"

# Stosujemy poziomy ryzyka TYLKO dla klientów z prediction_label = Yes
results['risk_level'] = results.apply(
    lambda row: get_risk_level(row['prediction_score']) if row['prediction_label'] == 'Yes' else 'LOW',
    axis=1
)
results['recommended_action'] = results.apply(
    lambda row: get_action(row['prediction_score']) if row['prediction_label'] == 'Yes' else 'Monitoring',
    axis=1
)

# ============================================================================
# WYŚWIETLENIE WYNIKÓW
# ============================================================================

print("\n" + "="*80)
print("📊 WYNIKI PREDYKCJI")
print("="*80)

# Statystyki
churn_yes = (results['prediction_label'] == 'Yes').sum()
churn_no = (results['prediction_label'] == 'No').sum()

print(f"\n👥 Liczba klientów: {len(results)}")
print(f"🔴 Przewidywane odejścia: {churn_yes} ({churn_yes/len(results)*100:.1f}%)")
print(f"🟢 Przewidywane pozostanie: {churn_no} ({churn_no/len(results)*100:.1f}%)")

# Klienci wymagający uwagi (prediction_label = Yes)
at_risk = results[results['prediction_label'] == 'Yes']

if len(at_risk) > 0:
    print(f"\n⚠️ KLIENCI WYMAGAJĄCY UWAGI: {len(at_risk)}\n")
    
    for idx, row in at_risk.iterrows():
        print(f"👤 {row['customerID']}")
        print(f"   Prawdopodobieństwo odejścia: {row['prediction_score']:.2%}")
        print(f"   Ryzyko: {row['risk_level']}")
        print(f"   Akcja: {row['recommended_action']}")
        print()
else:
    print("\n✅ Brak klientów z wysokim ryzykiem")

# ============================================================================
# ZAPIS WYNIKÓW
# ============================================================================

print("💾 Zapisywanie wyników...")

# Pełne wyniki
predictions.to_csv('data/predictions_results.csv', index=False)
print("✅ Zapisano: data/predictions_results.csv (pełne dane)")

# Podsumowanie
results.to_csv('data/predictions_summary.csv', index=False)
print("✅ Zapisano: data/predictions_summary.csv (podsumowanie)")

# ============================================================================
# PODSUMOWANIE
# ============================================================================

print("\n" + "="*80)
print("🎉 PREDYKCJE ZAKOŃCZONE!")
print("="*80)
print("\n📦 Pliki wyjściowe:")
print("   - data/predictions_results.csv (pełne dane)")
print("   - data/predictions_summary.csv (podsumowanie + rekomendacje)")
print("\n🚀 Wyniki gotowe do użycia!")
print("="*80)
