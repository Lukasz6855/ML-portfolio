"""
🎯 Threshold - Balansowanie Recall vs Precision w Churn Prediction

Cel: Pokazać jak threshold wpływa na trade-off między wykrywaniem
     klientów a liczbą fałszywych alarmów.
"""

# Import bibliotek
import pandas as pd
from pycaret.classification import *
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

print("✅ Biblioteki zaimportowane!")

# Wczytanie i przygotowanie danych
df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

print(f"\n📊 Liczba klientów: {len(df)}")
print(f"📋 Liczba kolumn: {len(df.columns)}")

# Przygotowanie danych
df = df.drop('customerID', axis=1)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(0, inplace=True)

print("\n📊 Rozkład churn:")
print(df['Churn'].value_counts())
print(f"\nProcent odchodzących: {(df['Churn'] == 'Yes').sum() / len(df) * 100:.1f}%")

# Konfiguracja PyCaret
clf_setup = setup(
    data=df,
    target='Churn',
    session_id=123,
    train_size=0.8,
    fold=5,
    normalize=True,
    verbose=False
)

print("\n✅ PyCaret skonfigurowany!")

# Trening modelu z optymalizacją pod Recall
print("\n🔄 Porównywanie modeli z optymalizacją pod RECALL...\n")

best_model = compare_models(sort='Recall', n_select=1)

print("\n✅ Najlepszy model (pod kątem Recall) wybrany!")

# Tworzenie finalizowanego modelu
print("\n📊 Tworzenie modelu...")
final_model = create_model(best_model)

# Wyświetlanie confusion matrix
print("\n📊 Wyświetlanie Confusion Matrix...")
plot_model(final_model, plot='confusion_matrix')

# ============================================================================
# TUNING MODELU POD KĄTEM RECALL
# ============================================================================
print("\n" + "="*80)
print("🔧 TUNING - Optymalizacja modelu pod kątem RECALL")
print("="*80)

# Zapisujemy metryki modelu bazowego
baseline_metrics = pull()
baseline_recall = baseline_metrics.loc['Mean', 'Recall']
baseline_precision = baseline_metrics.loc['Mean', 'Prec.']
baseline_accuracy = baseline_metrics.loc['Mean', 'Accuracy']
baseline_auc = baseline_metrics.loc['Mean', 'AUC']

print(f"\n📈 Model bazowy (przed tuningiem):")
print(f"   Recall:    {baseline_recall:.4f}")
print(f"   Precision: {baseline_precision:.4f}")
print(f"   Accuracy:  {baseline_accuracy:.4f}")
print(f"   AUC:       {baseline_auc:.4f}")

# Tunujemy model pod kątem Recall
print("\nSzukamy najlepszych hiperparametrów...")
tuned_model = tune_model(final_model, optimize='Recall', n_iter=20)

# Pobieramy metryki po tuningu
tuned_metrics = pull()
tuned_recall = tuned_metrics.loc['Mean', 'Recall']
tuned_precision = tuned_metrics.loc['Mean', 'Prec.']
tuned_accuracy = tuned_metrics.loc['Mean', 'Accuracy']
tuned_auc = tuned_metrics.loc['Mean', 'AUC']

print(f"\n📈 Model po tuningu:")
print(f"   Recall:    {tuned_recall:.4f}")
print(f"   Precision: {tuned_precision:.4f}")
print(f"   Accuracy:  {tuned_accuracy:.4f}")
print(f"   AUC:       {tuned_auc:.4f}")

# Porównanie i decyzja
recall_diff = (tuned_recall - baseline_recall) * 100

print(f"\n⚖️ Zmiana Recall: {recall_diff:+.2f} p.p.")

THRESHOLD_FOR_IMPROVEMENT = 1.0  # 1 punkt procentowy

if recall_diff >= THRESHOLD_FOR_IMPROVEMENT:
    print(f"✅ TUNING OPŁACALNY! Używamy tuned_model")
    selected_model = tuned_model
    model_name = "TUNED"
else:
    print(f"❌ TUNING NIE WART ZACHODU! Używamy modelu bazowego")
    selected_model = final_model
    model_name = "BAZOWY"

print(f"\n💡 Wybrany model: {model_name}")
print("="*80)

# Pobieranie danych testowych
X_test = get_config('X_test')
y_test = get_config('y_test')
y_test_numeric = (y_test == 'Yes').astype(int)

# Przewidywania (używamy wybranego modelu)
print(f"\n🤖 Używamy modelu: {model_name}")
predictions = predict_model(selected_model, data=X_test)
y_proba = predictions['prediction_score'].values

print(f"\n✅ Pobrano {len(X_test)} przykładów testowych")

# Funkcja do obliczania confusion matrix
def calculate_confusion_matrix(y_true, y_proba, threshold):
    """Oblicza confusion matrix dla danego threshold."""
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    return {
        'threshold': threshold,
        'confusion_matrix': cm,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp,
        'recall': recall,
        'precision': precision,
        'accuracy': accuracy
    }

# Testowanie różnych thresholdów
thresholds = [0.3, 0.5, 0.7]
results = []

for threshold in thresholds:
    result = calculate_confusion_matrix(y_test_numeric, y_proba, threshold)
    results.append(result)

print("\n✅ Obliczono confusion matrix dla wszystkich thresholdów!")

# Wizualizacja
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, result in enumerate(results):
    cm = result['confusion_matrix']
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['Zostaje', 'Odchodzi'],
                yticklabels=['Zostaje', 'Odchodzi'])
    
    axes[idx].set_title(
        f"Threshold = {result['threshold']}\n"
        f"Recall: {result['recall']:.2%} | Precision: {result['precision']:.2%}",
        fontsize=12, fontweight='bold'
    )
    
    axes[idx].set_ylabel('Rzeczywistość')
    axes[idx].set_xlabel('Przewidywanie')

plt.tight_layout()
plt.show()

print("\n📊 Confusion Matrix dla różnych thresholdów wyświetlona!")

# Szczegółowe porównanie
print("\n" + "="*80)
print("📊 SZCZEGÓŁOWE PORÓWNANIE THRESHOLDÓW")
print("="*80)

for result in results:
    print(f"\n{'='*80}")
    print(f"🎯 THRESHOLD = {result['threshold']} ({result['threshold']*100:.0f}%)")
    print(f"{'='*80}")
    
    print(f"\n📈 CONFUSION MATRIX:")
    print(f"   ✅ True Negative (TN):  {result['tn']:4d} - Prawidłowo: 'klient zostaje'")
    print(f"   ⚠️  False Positive (FP): {result['fp']:4d} - Fałszywy alarm")
    print(f"   ❌ False Negative (FN): {result['fn']:4d} - Przegapiony")
    print(f"   ✅ True Positive (TP):  {result['tp']:4d} - Prawidłowo wykryty")
    
    print(f"\n📊 METRYKI:")
    print(f"   Recall:    {result['recall']:.2%}")
    print(f"   Precision: {result['precision']:.2%}")
    print(f"   Accuracy:  {result['accuracy']:.2%}")

print(f"\n{'='*80}")

# Analiza biznesowa
cost_false_positive = 20  # zł
cost_false_negative = 500  # zł
retention_rate = 0.30
value_retained_customer = 500  # zł
cost_retention_attempt = 50  # zł

print("\n" + "="*80)
print("💰 ANALIZA BIZNESOWA")
print("="*80)

print(f"\n📋 Założenia:")
print(f"   - Koszt fałszywego alarmu (FP): {cost_false_positive} zł")
print(f"   - Koszt przegapienia (FN): {cost_false_negative} zł")
print(f"   - Koszt retencji: {cost_retention_attempt} zł")
print(f"   - Skuteczność retencji: {retention_rate*100:.0f}%")

for result in results:
    print(f"\n{'='*80}")
    print(f"🎯 THRESHOLD = {result['threshold']}")
    print(f"{'='*80}")
    
    cost_fp = result['fp'] * cost_false_positive
    cost_fn = result['fn'] * cost_false_negative
    cost_tp = result['tp'] * cost_retention_attempt
    revenue_tp = result['tp'] * retention_rate * value_retained_customer
    
    total_cost = cost_fp + cost_fn + cost_tp
    total_revenue = revenue_tp
    net_profit = total_revenue - total_cost
    
    print(f"\n💸 KOSZTY:")
    print(f"   Fałszywe alarmy (FP={result['fp']}): {cost_fp:,} zł")
    print(f"   Przegapieni (FN={result['fn']}): {cost_fn:,} zł")
    print(f"   Próby retencji (TP={result['tp']}): {cost_tp:,} zł")
    print(f"   SUMA: {total_cost:,} zł")
    
    print(f"\n💰 PRZYCHODY:")
    print(f"   Zatrzymani ({result['tp']*retention_rate:.0f} z {result['tp']}): {revenue_tp:,} zł")
    
    print(f"\n📊 BILANS:")
    if net_profit >= 0:
        print(f"   ✅ ZYSK: {net_profit:,} zł")
    else:
        print(f"   ❌ STRATA: {abs(net_profit):,} zł")

print(f"\n{'='*80}")

# Wnioski
print("\n" + "="*80)
print("🎯 WNIOSKI")
print("="*80)

print("""
📊 TRADE-OFF RECALL vs PRECISION:

1️⃣ THRESHOLD 0.3 (Liberalny):
   ✅ Wysoki Recall - wykrywamy więcej odchodzących
   ❌ Niski Precision - więcej fałszywych alarmów
   💡 Używaj gdy: Koszt przegapienia >> Koszt fałszywego alarmu

2️⃣ THRESHOLD 0.5 (Standardowy):
   ⚖️ Balans między Recall a Precision
   💡 Rozsądny kompromis

3️⃣ THRESHOLD 0.7 (Konserwatywny):
   ✅ Wysoki Precision - mało fałszywych alarmów
   ❌ Niski Recall - przegapiamy więcej klientów
   💡 Używaj gdy: Koszt fałszywego alarmu >> Koszt przegapienia

💡 W churn prediction zazwyczaj LEPIEJ:
   - Niższy threshold (0.3-0.4)
   - Wysoki Recall (wykryć więcej)
   - Niższy Precision (tolerować fałszywe alarmy)
   
   DLACZEGO? Koszt przegapienia (500 zł) >> Koszt alarmu (20 zł)
   
   ANALOGIA: Wykrywacz dymu - wolisz 10 fałszywych alarmów
            niż przegapić prawdziwy pożar!
""")

# Analiza rozkładu prawdopodobieństw
print("\n" + "="*80)
print("⚠️ ROZKŁAD PRAWDOPODOBIEŃSTW")
print("="*80)

count_below_03 = (y_proba < 0.3).sum()
count_03_05 = ((y_proba >= 0.3) & (y_proba < 0.5)).sum()
count_05_07 = ((y_proba >= 0.5) & (y_proba < 0.7)).sum()
count_above_07 = (y_proba >= 0.7).sum()

print(f"\n  < 0.3:  {count_below_03} klientów")
print(f"  0.3-0.5: {count_03_05} klientów")
print(f"  0.5-0.7: {count_05_07} klientów")
print(f"  ≥ 0.7:  {count_above_07} klientów")

if count_03_05 == 0 and count_below_03 == 0:
    print("\n💡 Model ma POLARYZOWANY rozkład (wszystkie ≥ 0.5)")
    print("   → Threshold 0.3 i 0.5 dają IDENTYCZNE wyniki!")
    print("   → Zostaw threshold 0.5 (domyślny)")

print("="*80)
print("🎉 ANALIZA ZAKOŃCZONA!")
print("="*80)
