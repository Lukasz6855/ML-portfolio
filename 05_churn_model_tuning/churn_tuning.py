"""
🎯 Tuning Modelu - Czy Optymalizacja Ma Sens Biznesowy?

Cel projektu:
- Wytrenować model przewidujący churn klientów
- Porównać model "z pudełka" vs model po tuningu
- Ocenić, czy tuning daje realną wartość biznesową
"""

# Import bibliotek
import pandas as pd
from pycaret.classification import *
import numpy as np

print("✅ Biblioteki zaimportowane!")

# Wczytanie danych
df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

print(f"\n📊 Liczba klientów: {len(df)}")
print(f"📋 Liczba kolumn: {len(df.columns)}")
print("\n🔍 Pierwsze 5 wierszy danych:")
print(df.head())

# Przygotowanie danych
df = df.drop('customerID', axis=1)

print("\n🔍 Sprawdzanie brakujących wartości...")
missing = df.isnull().sum()
print(f"\nLiczba brakujących wartości: {missing.sum()}")

# Naprawa kolumny TotalCharges
print("\n🔧 Naprawiamy kolumnę TotalCharges...")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
nan_count = df['TotalCharges'].isnull().sum()
print(f"Znaleziono {nan_count} nieprawidłowych wartości w TotalCharges")
if nan_count > 0:
    df['TotalCharges'].fillna(0, inplace=True)
    print("✅ Wypełniono zerami (nowi klienci bez historii płatności)")
print("\n📊 Rozkład targetu (Churn):")
print(df['Churn'].value_counts())
print(f"\nProcent klientów, którzy odeszli: {(df['Churn'] == 'Yes').sum() / len(df) * 100:.1f}%")

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

print("\n✅ PyCaret skonfigurowany i gotowy do pracy!")

# Porównanie modeli i wybór najlepszego
print("\n🔄 Porównywanie modeli...\n")
best_model = compare_models(sort='AUC', n_select=1)

print("\n✅ Najlepszy model wybrany!")

# Zapisanie wyników PRZED tuningiem
print("\n📊 Trenowanie i zapisywanie wyników PRZED tuningiem...\n")

model_before = create_model(best_model, fold=5)
results_before = pull()
accuracy_before = results_before.loc['Mean', 'Accuracy']
auc_before = results_before.loc['Mean', 'AUC']
recall_before = results_before.loc['Mean', 'Recall']
precision_before = results_before.loc['Mean', 'Prec.']

print("\n✅ Wyniki PRZED tuningiem zapisane!")
print(f"\n📈 Kluczowe metryki PRZED tuningiem:")
print(f"   - Accuracy (dokładność ogólna): {accuracy_before:.4f}")
print(f"   - AUC (zdolność rozróżniania): {auc_before:.4f}")
print(f"   - Recall (% wykrytych odejść): {recall_before:.4f}")
print(f"   - Precision (% trafnych alertów): {precision_before:.4f}")

# Tuning modelu
print("\n🔧 Tuning modelu (20 iteracji)...\n")
model_after = tune_model(model_before, optimize='AUC', n_iter=20)

# Pobieramy wyniki PO tuningu
results_after = pull()

# Zapisujemy kluczowe metryki PO tuningu
accuracy_after = results_after.loc['Mean', 'Accuracy']
auc_after = results_after.loc['Mean', 'AUC']
recall_after = results_after.loc['Mean', 'Recall']
precision_after = results_after.loc['Mean', 'Prec.']

print("\n✅ Tuning zakończony!")
print(f"\n📈 Kluczowe metryki PO tuningu:")
print(f"   - Accuracy (dokładność ogólna): {accuracy_after:.4f}")
print(f"   - AUC (zdolność rozróżniania): {auc_after:.4f}")
print(f"   - Recall (% wykrytych odejść): {recall_after:.4f}")
print(f"   - Precision (% trafnych alertów): {precision_after:.4f}")

# Porównanie wyników PRZED vs PO tuningu
print("\n" + "="*60)
print("📊 PORÓWNANIE: MODEL PRZED vs PO TUNINGU")
print("="*60)

# Obliczamy różnice (w punktach procentowych)
accuracy_diff = (accuracy_after - accuracy_before) * 100
auc_diff = (auc_after - auc_before) * 100
recall_diff = (recall_after - recall_before) * 100
precision_diff = (precision_after - precision_before) * 100

# Wyświetlamy szczegółowe porównanie
print(f"\n1️⃣ ACCURACY (Dokładność ogólna):")
print(f"   Przed: {accuracy_before:.4f} ({accuracy_before*100:.2f}%)")
print(f"   Po:    {accuracy_after:.4f} ({accuracy_after*100:.2f}%)")
print(f"   Zmiana: {accuracy_diff:+.2f} punktów procentowych")

print(f"\n2️⃣ AUC (Zdolność rozróżniania klas):")
print(f"   Przed: {auc_before:.4f}")
print(f"   Po:    {auc_after:.4f}")
print(f"   Zmiana: {auc_diff:+.2f} punktów procentowych")

print(f"\n3️⃣ RECALL (Ile % odchodzących klientów wykrywamy):")
print(f"   Przed: {recall_before:.4f} ({recall_before*100:.2f}%)")
print(f"   Po:    {recall_after:.4f} ({recall_after*100:.2f}%)")
print(f"   Zmiana: {recall_diff:+.2f} punktów procentowych")

print(f"\n4️⃣ PRECISION (Ile % naszych alertów jest trafnych):")
print(f"   Przed: {precision_before:.4f} ({precision_before*100:.2f}%)")
print(f"   Po:    {precision_after:.4f} ({precision_after*100:.2f}%)")
print(f"   Zmiana: {precision_diff:+.2f} punktów procentowych")

print("\n" + "="*60)

# Analiza biznesowa
total_customers = 10000
churn_rate = 0.27
churning_customers = int(total_customers * churn_rate)
retention_cost = 50
customer_value = 500

print("\n" + "="*60)
print("💰 ANALIZA BIZNESOWA - CZY TUNING SIĘ OPŁACA?")
print("="*60)

print(f"\n📊 Założenia:")
print(f"   - Baza klientów: {total_customers:,}")
print(f"   - Klienci odchodzący: {churning_customers:,} ({churn_rate*100:.0f}%)")
print(f"   - Koszt próby zatrzymania: {retention_cost} zł")
print(f"   - Wartość klienta rocznie: {customer_value} zł")

detected_before = int(churning_customers * recall_before)
detected_after = int(churning_customers * recall_after)
additional_detected = detected_after - detected_before

print(f"\n🎯 Wykrywanie klientów:")
print(f"   - Przed tuningiem: {detected_before:,} klientów")
print(f"   - Po tuningu: {detected_after:,} klientów")
print(f"   - DODATKOWO wykrytych: {additional_detected:,} klientów")

retention_success_rate = 0.30
additional_retained = int(additional_detected * retention_success_rate)
additional_cost = additional_detected * retention_cost
additional_revenue = additional_retained * customer_value
net_benefit = additional_revenue - additional_cost

print(f"\n💼 Skutki biznesowe (przy 30% skuteczności retencji):")
print(f"   - Dodatkowo zatrzymanych klientów: {additional_retained:,}")
print(f"   - Dodatkowy koszt retencji: {additional_cost:,} zł")
print(f"   - Dodatkowy przychód (zatrzymani): {additional_revenue:,} zł")
if net_benefit >= 0:
    print(f"   - ZYSK NETTO Z TUNINGU: {net_benefit:,} zł rocznie")
else:
    print(f"   - STRATA NETTO Z TUNINGU: {abs(net_benefit):,} zł rocznie")

print("\n" + "="*60)

# Wnioski końcowe
print("\n" + "="*60)
print("🎯 WNIOSKI - CZY TUNING MA SENS BIZNESOWY?")
print("="*60)

if auc_diff > 0.5:
    print("\n✅ TUNING DAŁ REALNĄ POPRAWĘ TECHNICZNĄ!")
    print(f"   AUC wzrósł o {auc_diff:.2f} punktów procentowych")
    print(f"   To znacząca poprawa zdolności modelu do rozróżniania klientów")
elif auc_diff > 0:
    print("\n⚠️ TUNING DAŁ NIEWIELKĄ POPRAWĘ TECHNICZNĄ")
    print(f"   AUC wzrósł o {auc_diff:.2f} punktów procentowych")
    print(f"   Poprawa jest minimalna, model niewiele zyskał")
else:
    print("\n❌ TUNING NIE POPRAWIŁ MODELU")
    print(f"   AUC zmienił się o {auc_diff:.2f} punktów procentowych")
    print(f"   Model nie zyskał na tuningu")

if recall_diff > 1.0:
    print("\n✅ WYKRYWAMY ZNACZNIE WIĘCEJ ODCHODZĄCYCH KLIENTÓW!")
    print(f"   Recall wzrósł o {recall_diff:.2f} punktów procentowych")
    print(f"   Wykrywamy {additional_detected:,} więcej klientów zagrożonych odejściem")
elif recall_diff > 0:
    print("\n✅ WYKRYWAMY TROCHĘ WIĘCEJ ODCHODZĄCYCH KLIENTÓW")
    print(f"   Recall wzrósł o {recall_diff:.2f} punktów procentowych")
    print(f"   Wykrywamy {additional_detected:,} więcej klientów zagrożonych odejściem")
else:
    print("\n⚠️ NIE WYKRYWAMY WIĘCEJ KLIENTÓW")
    print(f"   Recall zmienił się o {recall_diff:.2f} punktów procentowych")

print("\n💰 ANALIZA FINANSOWA:")
if net_benefit > 10000:
    print(f"   ✅ TUNING MA DUŻY SENS BIZNESOWY!")
    print(f"   Roczny zysk: {net_benefit:,} zł")
    print(f"   ROI: {(net_benefit/additional_cost)*100:.0f}% (świetny zwrot z inwestycji!)")
elif net_benefit > 0:
    print(f"   ✅ TUNING MA SENS BIZNESOWY")
    print(f"   Roczny zysk: {net_benefit:,} zł")
    print(f"   ROI: {(net_benefit/additional_cost)*100:.0f}% (opłaca się!)")
else:
    print(f"   ❌ TUNING NIE MA SENSU BIZNESOWEGO")
    print(f"   Strata: {abs(net_benefit):,} zł rocznie")
    print(f"   Koszt retencji przewyższa korzyści")

print("\n🎯 REKOMENDACJA:")
if net_benefit > 5000 and auc_diff > 0.3:
    print("   🌟 ZDECYDOWANIE WARTO WDROŻYĆ MODEL PO TUNINGU!")
    print("   Tuning dał znaczącą poprawę i generuje solidny zysk")
    print("   Model będzie generował dodatkowy zysk")
elif net_benefit > 0 and auc_diff > 0:
    print("   ✅ WARTO WDROŻYĆ MODEL PO TUNINGU")
    print("   Tuning dał niewielką, ale pozytywną poprawę")
    print("   Model będzie generował dodatkowy zysk")
else:
    print("   ⚠️ ZOSTAŃ PRZY MODELU PODSTAWOWYM")
    print("   Tuning nie dał poprawy lub okazał się nawet gorszy")

print("\n" + "="*60)

# Zapisanie najlepszego modelu (przed tuningiem - bo okazał się lepszy!)
print("\n💾 Zapisywanie modelu bazowego (najlepszego)...")
save_model(model_before, 'models/churn_best_model')

print("✅ Model zapisany w folderze 'models/churn_best_model'!")
print("📝 Zapisaliśmy model PRZED tuningiem, bo okazał się lepszy!")
print("📝 Możesz go później wczytać używając: load_model('models/churn_best_model')")

# Podsumowanie
print("\n" + "="*60)
print("📚 PODSUMOWANIE")
print("="*60)
print(f"\n- Recall: {recall_after*100:.0f}% (wykrywamy {int(recall_after*100)}/100 odchodzących klientów)")
print(f"- Precision: {precision_after*100:.0f}% ({int(precision_after*100)}/100 alertów trafnych)")

print("\n" + "="*60)
print("🎉 ANALIZA ZAKOŃCZONA!")
print("="*60)
