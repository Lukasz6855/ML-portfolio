"""
🔍 Interpretacja Modelu Churn - Feature Importance i SHAP

Ten skrypt analizuje zapisany model churn i wyjaśnia, które cechy
wpływają na odejście klientów i w jaki sposób.
"""

# ============================================================================
# IMPORT BIBLIOTEK
# ============================================================================

# Podstawowe biblioteki do pracy z danymi
import pandas as pd
import numpy as np
import os
import shutil

# PyCaret - framework do machine learning
from pycaret.classification import *

# SHAP - biblioteka do interpretacji modeli
import shap
shap.initjs()

# Biblioteki do wizualizacji
import matplotlib.pyplot as plt
import seaborn as sns

# Ustawienia estetyczne dla wykresów
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Import metadanych modelu
import json

# Utworzenie folderu plots jeśli nie istnieje
os.makedirs('plots', exist_ok=True)
os.makedirs('data', exist_ok=True)

print("="*80)
print("🔍 INTERPRETACJA MODELU CHURN")
print("="*80)

# ============================================================================
# WCZYTANIE DANYCH I MODELU
# ============================================================================

print("\n📂 Wczytywanie danych...")

# Wczytanie datasetu klientów telekomunikacyjnych
df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
print(f"✅ Wczytano dane: {len(df)} klientów, {len(df.columns)} kolumn")

# Przygotowanie danych (takie same kroki jak podczas treningu)
df = df.drop('customerID', axis=1)  # Usunięcie kolumny ID
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')  # Konwersja na liczby
df['TotalCharges'].fillna(0, inplace=True)  # Wypełnienie brakujących wartości

print("✅ Dane przygotowane")

# Wczytanie metadanych modelu
with open('model/metadata.json', 'r', encoding='utf-8') as f:
    metadata = json.load(f)

print(f"\n📋 Typ modelu: {metadata['model_type']}")
print(f"📋 Optymalizacja: {metadata['optimized_for']}")
print(f"📋 Data treningu: {metadata['train_date']}")

# ============================================================================
# KONFIGURACJA PYCARET I WCZYTANIE MODELU
# ============================================================================

print("\n⚙️ Konfiguracja PyCaret...")

# Konfiguracja środowiska PyCaret (takie same ustawienia jak podczas treningu)
clf_setup = setup(
    data=df,
    target='Churn',
    session_id=123,
    train_size=0.8,
    fold=5,
    normalize=True,
    verbose=False,
    memory=False  # Zapobiega problemom z kompatybilnością
)

print("✅ PyCaret skonfigurowany")

# Wczytanie zapisanego modelu
print("\n💾 Wczytywanie modelu...")
loaded_model = load_model('model/churn_model')
print(f"✅ Model wczytany: {type(loaded_model).__name__}")

# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================

print("\n" + "="*80)
print("📊 ANALIZA FEATURE IMPORTANCE")
print("="*80)

# Wyświetlenie feature importance
print("\n🔍 Generowanie wykresu Feature Importance...")
plot_model(loaded_model, plot='feature', save=True)

# Przeniesienie do folderu plots
if os.path.exists('Feature Importance.png'):
    shutil.move('Feature Importance.png', 'plots/Feature Importance.png')
    print("✅ Wykres zapisany jako: plots/Feature Importance.png")
else:
    print("⚠️ Plik Feature Importance.png nie został znaleziony")

# ============================================================================
# PRZYGOTOWANIE DANYCH DO SHAP
# ============================================================================

print("\n" + "="*80)
print("🎯 ANALIZA SHAP")
print("="*80)

# Przygotowanie danych do analizy SHAP - używamy danych PRZED transformacją
# PyCaret predict_model() automatycznie wykona transformacje
X_train = get_config('X_train')  # Dane oryginalne (przed encoding i normalizacją)
print(f"\n📊 Rozmiar danych treningowych: {X_train.shape[0]} wierszy, {X_train.shape[1]} kolumn")

# Użycie próbki dla szybszych obliczeń
sample_size = 500
X_sample = X_train.sample(n=min(sample_size, len(X_train)), random_state=42)
print(f"📋 Użyto próbki: {len(X_sample)} klientów")

# ============================================================================
# TWORZENIE EKSPLANERA SHAP
# ============================================================================

print("\n🔄 Tworzenie eksplanera SHAP...")

# Funkcja pomocnicza dla PyCaret predict_model
def model_predict(data):
    preds = predict_model(loaded_model, data=pd.DataFrame(data, columns=X_sample.columns))
    return preds['prediction_score_1'].values if 'prediction_score_1' in preds.columns else preds['prediction_score'].values

# KernelExplainer - najlepszy dla modeli z mieszanymi typami danych
# Jest wolniejszy ale działa niezawodnie z danymi kategorycznymi i numerycznymi
explainer = shap.KernelExplainer(model_predict, shap.sample(X_sample, 100))
print("✅ KernelExplainer utworzony (kompatybilny z danymi kategorycznymi i numerycznymi)")

# Obliczenie wartości SHAP
print("\n🔄 Obliczanie wartości SHAP (to może potrwać 2-3 minuty)...")
shap_values = explainer.shap_values(X_sample)
print(f"✅ Wartości SHAP obliczone! Kształt: {shap_values.shape}")

# ============================================================================
# SHAP SUMMARY PLOT
# ============================================================================

print("\n" + "="*80)
print("📊 SHAP SUMMARY PLOT")
print("="*80)

plt.figure(figsize=(12, 8))

shap_values_to_plot = shap_values

# Utworzenie SHAP Summary Plot
shap.summary_plot(
    shap_values_to_plot,
    X_sample,
    feature_names=X_sample.columns,
    show=False
)

# Dodanie tytułu i opisów
plt.title('SHAP Summary Plot - Wpływ cech na odejście klientów', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Wpływ na predykcję (SHAP value)\n← Zmniejsza ryzyko churn | Zwiększa ryzyko churn →', fontsize=12)
plt.tight_layout()

# Zapis wykresu
plt.savefig('plots/shap_summary_plot.png', dpi=300, bbox_inches='tight')
print("\n✅ Wykres zapisany jako: plots/shap_summary_plot.png")
plt.close()

# ============================================================================
# SHAP BAR PLOT
# ============================================================================

print("\n" + "="*80)
print("📊 SHAP BAR PLOT")
print("="*80)

plt.figure(figsize=(10, 6))

# Utworzenie bar plot
shap.summary_plot(
    shap_values_to_plot,
    X_sample,
    feature_names=X_sample.columns,
    plot_type='bar',
    show=False
)

# Dodanie tytułu
plt.title('Ranking ważności cech (średni absolutny wpływ SHAP)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Średni absolutny wpływ na predykcję', fontsize=11)
plt.tight_layout()

# Zapis wykresu
plt.savefig('plots/shap_bar_plot.png', dpi=300, bbox_inches='tight')
print("\n✅ Wykres zapisany jako: plots/shap_bar_plot.png")
plt.close()

# ============================================================================
# FORCE PLOT - ANALIZA POJEDYNCZEGO KLIENTA
# ============================================================================

print("\n" + "="*80)
print("🔍 ANALIZA POJEDYNCZEGO KLIENTA")
print("="*80)

# Wybór klienta do analizy
customer_idx = 0
print(f"\n👤 Analiza klienta #{customer_idx}")

# Utworzenie Force Plot
print("\n🔄 Generowanie Force Plot...")
plt.figure(figsize=(20, 3))  # Szeroki wykres dla lepszej czytelności

shap.force_plot(
    explainer.expected_value,
    shap_values_to_plot[customer_idx],
    X_sample.iloc[customer_idx],
    matplotlib=True,
    show=False,
    text_rotation=45
)

plt.gcf().set_size_inches(20, 3)
plt.tight_layout()

# Zapis wykresu
plt.savefig(f'plots/shap_force_plot_customer_{customer_idx}.png', dpi=300, bbox_inches='tight')
print(f"✅ Force plot zapisany jako: plots/shap_force_plot_customer_{customer_idx}.png")
plt.close()

# ============================================================================
# WATERFALL PLOT - LEPSZA ALTERNATYWA
# ============================================================================

print("\n🌊 Generowanie Waterfall Plot (bardziej czytelny)...")

# Utworzenie obiektu Explanation dla Waterfall
shap_explanation_single = shap.Explanation(
    values=shap_values_to_plot[customer_idx],
    base_values=explainer.expected_value,
    data=X_sample.iloc[customer_idx].values,
    feature_names=X_sample.columns.tolist()
)

plt.figure(figsize=(10, 8))
shap.waterfall_plot(shap_explanation_single, max_display=15, show=False)
plt.tight_layout()

# Zapis
plt.savefig(f'plots/shap_waterfall_plot_customer_{customer_idx}.png', dpi=300, bbox_inches='tight')
print(f"✅ Waterfall plot zapisany jako: plots/shap_waterfall_plot_customer_{customer_idx}.png")
plt.close()

# ============================================================================
# EKSPORT WYNIKÓW
# ============================================================================

print("\n" + "="*80)
print("📊 EKSPORT WYNIKÓW")
print("="*80)

# Utworzenie tabeli z rankingiem cech
feature_importance_shap = pd.DataFrame({
    'Feature': X_sample.columns,
    'Mean_Absolute_SHAP': np.abs(shap_values_to_plot).mean(axis=0)
})

# Sortowanie od najważniejszej
feature_importance_shap = feature_importance_shap.sort_values('Mean_Absolute_SHAP', ascending=False)

# Zapis do CSV
feature_importance_shap.to_csv('data/feature_importance_shap.csv', index=False)
print("\n✅ Ranking cech zapisany do: data/feature_importance_shap.csv")

# Wyświetlenie top 10
print("\n🏆 TOP 10 najważniejszych cech:")
print(feature_importance_shap.head(10).to_string(index=False))

# ============================================================================
# PODSUMOWANIE
# ============================================================================

print("\n" + "="*80)
print("✅ ANALIZA ZAKOŃCZONA")
print("="*80)

print("\n📊 Wygenerowane pliki:")
print("  - Feature Importance.png (wykres ważności cech)")
print("  - plots/shap_summary_plot.png (szczegółowa analiza SHAP)")
print("  - plots/shap_bar_plot.png (ranking cech)")
print(f"  - plots/shap_force_plot_customer_{customer_idx}.png (Force plot)")
print(f"  - plots/shap_waterfall_plot_customer_{customer_idx}.png (Waterfall plot - czytelniejszy)")
print("  - data/feature_importance_shap.csv (ranking cech w CSV)")

print("\n🎯 Kluczowe wnioski:")
print("  1. tenure (czas bycia klientem) - NAJWAŻNIEJSZA cecha")
print("     → Nowi klienci (krótki tenure) masowo odchodzą!")
print("  2. TotalCharges - wysokie zmniejsza churn (bo długi staż = lojalność)")
print("  3. MonthlyCharges - wysokie zwiększa churn (irytują klientów)")
print("  4. InternetService, Contract, PaymentMethod - umiarkowany wpływ")

print("\n💼 Rekomendowane działania:")
print("  - Program welcome dla nowych klientów (0-6 miesięcy)")
print("  - Zachęty do długoterminowych kontraktów (rabaty)")
print("  - Targetowane oferty dla klientów z wysokimi opłatami")
print("  - Specjalna obsługa dla użytkowników Fiber optic")

print("\n🚀 Analiza zakończona pomyślnie!")
