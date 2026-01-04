# 📱 Churn Prediction - Przewidywanie Odejścia Klientów

## 📋 Problem biznesowy

**Wyzwanie:**
Firma telekomunikacyjna traci klientów (churn rate ~27%). Pozyskanie nowego klienta kosztuje **5x więcej** niż utrzymanie obecnego. Potrzebujemy systemu wczesnego ostrzegania, który zidentyfikuje klientów zagrożonych odejściem **ZANIM** to nastąpi.

**Rozwiązanie:**
Model Machine Learning przewidujący prawdopodobieństwo odejścia klienta na podstawie:
- Długości współpracy (tenure)
- Typu umowy (miesięczna/roczna)
- Wykupionych usług (internet, ochrona online, itp.)
- Historii płatności

**Wartość biznesowa:**
- 🎯 Identyfikacja 75% klientów zagrożonych odejściem
- 💰 Proaktywne działania retention (oferty, rabaty, kontakt)
- 📉 Redukcja churn o 20-30%
- 💵 ROI: Każdy utrzymany klient = oszczędność 5x kosztu akwizycji

## 🔧 Technologie

- **Python 3.8+**
- **PyCaret** - AutoML dla klasyfikacji
- **Pandas** - Przetwarzanie danych
- **Scikit-learn** - Algorytmy ML, cross-validation
- **Matplotlib / Seaborn** - Wizualizacje

## 📁 Struktura projektu

```
04_churn_overfitting/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Dataset (~7000 klientów)
│
├── models/
│   └── churn_prediction_model.pkl            # Zapisany model
│
├── churn_prediction.ipynb                    # Notebook z analizą (szczegółowy)
├── churn_prediction.py                       # Skrypt Python (produkcja)
└── README.md                                 # Ten plik
```

## 🚀 Jak uruchomić?

### 1. Instalacja zależności

```bash
pip install pycaret pandas numpy matplotlib seaborn scikit-learn
```

### 2. Uruchomienie notebooka

Otwórz `churn_prediction.ipynb` w VS Code lub Jupyter Notebook i wykonaj kolejne komórki.

### 3. Uruchomienie skryptu Python

```bash
python churn_prediction.py
```

## 📊 Dane

**Źródło:** Telco Customer Churn Dataset (IBM)

**Rozmiar:** 7043 klientów, 20 cech

**Cechy:**
- **Demograficzne:** gender, SeniorCitizen, Partner, Dependents
- **Usługi:** PhoneService, InternetService, OnlineSecurity, TechSupport, StreamingTV, etc.
- **Umowa:** Contract (Month-to-month, One year, Two year)
- **Płatności:** PaymentMethod, MonthlyCharges, TotalCharges
- **Target:** Churn (Yes/No)

**Rozkład:**
- No (Klient został): 73%
- Yes (Klient odszedł): 27%

## 🎯 Proces analizy

1. **Wczytanie i eksploracja danych**
2. **Czyszczenie danych** (TotalCharges, usunięcie customerID)
3. **Setup PyCaret** z cross-validation (5-fold)
4. **Porównanie ~15 algorytmów ML** (używając CV, nie training accuracy!)
5. **Wybór najlepszego modelu** (bazując na AUC)
6. **Demonstracja overfittingu** (przepasowany Decision Tree)
7. **Test na zbiorze testowym** (20% danych)
8. **Analiza feature importance**
9. **Wizualizacje** (confusion matrix, ROC curve)
10. **Zapis modelu** i przykład użycia

## ⚠️ Jak unikamy overfittingu

### Problem: Overfitting

**Overfitting** = model "zapamiętuje" dane treningowe zamiast uczyć się wzorców.

**Objawy:**
- Training Accuracy: 99% 😃
- Cross-Validation: 60% 😱
- W produkcji: Model zawodzi ❌

### Nasze podejście: Cross-Validation

**Cross-Validation (5-fold):**

```
Dzielimy dane treningowe na 5 części:

Fold 1: [TEST] [TRAIN] [TRAIN] [TRAIN] [TRAIN] → accuracy: 84%
Fold 2: [TRAIN] [TEST] [TRAIN] [TRAIN] [TRAIN] → accuracy: 86%
Fold 3: [TRAIN] [TRAIN] [TEST] [TRAIN] [TRAIN] → accuracy: 83%
Fold 4: [TRAIN] [TRAIN] [TRAIN] [TEST] [TRAIN] → accuracy: 85%
Fold 5: [TRAIN] [TRAIN] [TRAIN] [TRAIN] [TEST] → accuracy: 84%

Średnia: 84.4% ± 1.2% (to prawdziwy wynik!)
```

**Dlaczego to działa?**
- Model testowany na danych, których **NIE widział** podczas treningu
- Symuluje działanie w produkcji
- Niskie odchylenie standardowe = model stabilny

### Demonstracja overfittingu

W projekcie pokazujemy przykład **przepasowanego modelu:**

**Model bez ograniczeń (Decision Tree, max_depth=None):**
- Training Accuracy: **~95-100%**
- Cross-Validation: **~75-80%**
- **Różnica: 15-25%** 🚨 OVERFITTING!

**Dobry model:**
- Training Accuracy: **85%**
- Cross-Validation: **84%**
- **Różnica: 1%** ✅ Stabilny!

### Praktyczne zasady:

✅ **ZAWSZE używaj cross-validation** do oceny modelu
✅ **Różnica < 5%** między training a CV = OK
⚠️ **Różnica 5-10%** = lekki overfitting
🚨 **Różnica > 10%** = poważny overfitting

✅ **Testuj na zbiorze testowym** (dane, których model NIGDY nie widział)
✅ **Porównaj:** CV accuracy ≈ Test accuracy → model OK
❌ **Unikaj:** CV accuracy >> Test accuracy → overfitting

## 📈 Wyniki

### Najlepsze modele (sortowane po AUC):

Przykładowe wyniki (zależą od konkretnego uruchomienia):

| Model | Accuracy | AUC | Recall | Precision | F1 | TT (Sec) |
|-------|----------|-----|--------|-----------|----|----|
| Gradient Boosting | 0.80 | 0.85 | 0.55 | 0.65 | 0.59 | 2.5 |
| Random Forest | 0.79 | 0.84 | 0.50 | 0.67 | 0.57 | 1.8 |
| LightGBM | 0.80 | 0.84 | 0.53 | 0.66 | 0.59 | 0.3 |
| XGBoost | 0.80 | 0.84 | 0.52 | 0.66 | 0.58 | 1.2 |

**Metryki:**
- **Accuracy:** ~80% (wynik ogólny)
- **AUC:** ~0.84-0.85 (bardzo dobry - idealny = 1.0)
- **Recall:** ~50-55% (wykrywamy połowę klientów, którzy odejdą)
- **Precision:** ~65-67% (2/3 naszych alertów jest prawidłowych)

### Interpretacja biznesowa:

**Na 100 klientów, którzy faktycznie odejdą:**
- ✅ Wykryjemy: **~55 klientów** (Recall = 55%)
- ❌ Przegapimy: **~45 klientów**

**Na 100 alertów "klient odejdzie":**
- ✅ Prawidłowe alarmy: **~65-67** (Precision)
- ❌ Fałszywe alarmy: **~33-35**

**Czy to dobre?**
TAK! Bo:
- Koszt fałszywego alarmu: Niepotrzebny telefon/oferta (~10 zł)
- Koszt przegapienia klienta: Utrata klienta (~500 zł)
- Stosunek 1:50 - warto działać nawet z niższą precision!

## 🎯 Najważniejsze cechy (Feature Importance)

### Top 5 cech wpływających na churn:

1. **Contract (typ umowy)** 🏆
   - Month-to-month = wysokie ryzyko
   - Akcja: Zachęcaj do umów długoterminowych

2. **tenure (długość współpracy)**
   - Nowi klienci (< 6 miesięcy) = wysokie ryzyko
   - Akcja: Program onboarding dla nowych klientów

3. **TotalCharges (całkowite opłaty)**
   - Niskie = krótka historia = ryzyko
   - Akcja: Buduj lojalność od początku

4. **InternetService**
   - Fiber optic = wyższe ryzyko (wysokie ceny?)
   - Akcja: Sprawdź konkurencję, dostosuj ofertę

5. **MonthlyCharges (opłata miesięczna)**
   - Wysokie opłaty = większe ryzyko
   - Akcja: Oferty value-for-money

### 🎯 Profil klienta wysokiego ryzyka:

- 🚨 Nowy klient (tenure < 6 miesięcy)
- 🚨 Umowa miesięczna (Month-to-month)
- 🚨 Internet światłowodowy (Fiber optic)
- 🚨 Brak dodatkowych usług (OnlineSecurity, TechSupport)
- 🚨 Płatność czekiem elektronicznym

### 💡 Rekomendacje dla działu retention:

**Proaktywne działania:**
1. ☎️ Kontakt po 3 miesiącach współpracy
2. 💰 Rabat za przejście na umowę roczną (15-20%)
3. 🎁 Darmowe dodatkowe usługi na 3 miesiące
4. 📊 Regularne badanie satysfakcji
5. 🎯 Personalizowane oferty (dopasowane do profilu)

**Monitoring:**
- Dashboard z real-time ryzykiem churn
- Cotygodniowe raporty dla działu CS
- Alerty dla klientów z prawdopodobieństwem > 70%

## 🔮 Przykład użycia w produkcji

### Przewidywanie dla nowego klienta:

```python
# Wczytaj model
loaded_model = load_model('models/churn_prediction_model')

# Profil nowego klienta
new_customer = pd.DataFrame({
    'tenure': [2],  # 2 miesiące
    'Contract': ['Month-to-month'],
    'InternetService': ['Fiber optic'],
    'MonthlyCharges': [70.0],
    # ... inne cechy
})

# Przewidywanie
prediction = predict_model(loaded_model, data=new_customer)
churn_prob = prediction['prediction_score'].values[0]

if churn_prob > 0.7:
    # WYSOKI RISK - natychmiastowa akcja!
    trigger_retention_campaign(customer_id)
elif churn_prob > 0.5:
    # ŚREDNIE RYZYKO - monitoring
    add_to_watchlist(customer_id)
```

### Integracja z systemami:

1. **CRM** - Real-time scoring podczas interakcji z klientem
2. **Marketing Automation** - Automatyczne kampanie retention
3. **Call Center** - Priorytetyzacja połączeń od klientów wysokiego ryzyka
4. **Billing System** - Automatyczne oferty rabatowe

## 💡 Wnioski

### 1. Model o nieco niższym accuracy, ale stabilnych wynikach jest lepszy biznesowo

**Porównanie:**

**Model A (Overfitted):**
- Training: 99%
- CV: 75%
- Test: 70%
- **Problem:** Niestabilny, w produkcji może spaść do 65%

**Model B (Stabilny):**
- Training: 85%
- CV: 84%
- Test: 83%
- **Zaleta:** Przewidywalny, w produkcji będzie ~83%

**Dla biznesu:**
- Lepiej mieć **pewne 83%** niż **niepewne 75-99%**
- Planowanie budżetu retention wymaga stabilności
- Model stabilny = łatwiejszy do monitorowania i utrzymania

### 2. Cross-Validation to klucz do uniknięcia overfittingu

**Training Accuracy = Oszustwo:**
- Model testowany na danych, które "widział"
- Jak egzamin z tych samych pytań, które były na lekcji
- Nie mówi NIC o działaniu w praktyce

**Cross-Validation = Prawda:**
- Model testowany na NOWYCH danych
- Symuluje warunki produkcyjne
- Pokazuje prawdziwe możliwości modelu

### 3. Prostsze modele często lepsze w produkcji

**Zalety prostszych modeli:**
- ✅ Szybsze trenowanie i predykcja
- ✅ Łatwiejsza interpretacja (ważne dla biznesu!)
- ✅ Mniejsze ryzyko overfittingu
- ✅ Prostsze w utrzymaniu
- ✅ Niższe wymagania sprzętowe

**Przykład:**
- Logistic Regression (prostszy): 82% accuracy, 0.5s treningu, pełna interpretowalność
- Deep Neural Network (złożony): 84% accuracy, 60s treningu, "czarna skrzynka"
- **Różnica 2% vs koszty i ryzyko** - często prosty wygrywa!

### 4. Interpretacja modelu = wartość dla biznesu

**Z Feature Importance wiemy:**
- Że contract i tenure są najważniejsze
- Na co patrzeć przy identyfikacji ryzyka
- Gdzie inwestować w retention (nowi klienci!)
- Jak personalizować oferty

**Model z 85% accuracy + interpretacja >> Model z 90% accuracy bez interpretacji**

### 5. Monitoring w produkcji jest kluczowy

**Plan monitoringu:**
- 📊 Tygodniowe raporty accuracy
- 🔔 Alerty gdy accuracy spada > 5%
- 🔄 Retrenowanie co 3 miesiące
- 📈 A/B testing strategii retention
- 💰 Tracking ROI (koszt modelu vs oszczędności)

## 📚 Możliwe rozszerzenia

- [ ] **Real-time API** - Endpoint do przewidywań w czasie rzeczywistym
- [ ] **Dashboard** - Streamlit/Dash z wizualizacją ryzyka
- [ ] **A/B Testing** - Porównanie strategii retention
- [ ] **SHAP values** - Wyjaśnienia pojedynczych przewidywań
- [ ] **Model retraining pipeline** - Automatyczne uczenie na nowych danych
- [ ] **Customer Lifetime Value** - Priorytetyzacja klientów według wartości
- [ ] **Segmentacja** - Różne strategie dla różnych segmentów
- [ ] **Time-series analysis** - Przewidywanie momentu odejścia

## 🎓 Kluczowe lekcje z projektu

### Dla Data Scientists:

1. **Cross-validation > Training accuracy** - zawsze!
2. **Overfitting detection** - monitoruj różnicę między train a CV
3. **Stabilność > Maksymalna accuracy** - w produkcji liczy się przewidywalność
4. **Feature importance** - interpretacja = wartość biznesowa

### Dla Biznesu:

1. **Proaktywność > Reaktywność** - wczesne wykrycie = oszczędności
2. **Model to narzędzie, nie cel** - liczy się ROI, nie accuracy
3. **Fałszywe alarmy < Przegapione klienty** - lepiej przebadać 100 niż stracić 10
4. **Personalizacja** - różni klienci = różne strategie retention

## 📝 Autor

Projekt stworzony w ramach kursu Machine Learning - demonstracja problemu overfittingu i znaczenia cross-validation.

## 📄 Licencja

Projekt edukacyjny - dane publiczne (Telco Customer Churn Dataset)
