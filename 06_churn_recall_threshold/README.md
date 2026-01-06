# 🎯 Projekt 6: Threshold Analysis - Recall vs Precision w Churn Prediction

## 📋 Opis projektu

Projekt analizuje **wpływ threshold (progu decyzyjnego)** na trade-off między **Recall a Precision** w problemie churn prediction. Pokazuje jak wybór progu wpływa na liczbę wykrytych klientów odchodzących vs liczbę fałszywych alarmów.

**Kluczowe pytania:**
1. Czy tuning modelu pod kątem Recall faktycznie poprawia wyniki?
2. Jak threshold wpływa na confusion matrix?
3. Który threshold jest najlepszy biznesowo?
4. Dlaczego w churn prediction często lepiej wybrać niższy threshold?

---

## 🎯 Cel analizy

- **Porównać model bazowy vs tuned** pod kątem Recall
- **Przeanalizować 3 thresholdy**: 0.3 (liberalny), 0.5 (standardowy), 0.7 (konserwatywny)
- **Obliczyć koszty biznesowe** dla różnych thresholdów
- **Wyjaśnić trade-off** między Recall a Precision w kontekście biznesowym

---

## 📊 Dataset

**Telco Customer Churn Dataset**
- Źródło: Kaggle / IBM Watson Analytics
- Liczba rekordów: 7,043 klientów
- Liczba cech: 19 (po usunięciu `customerID`)
- Target: `Churn` (Yes/No)
- Rozkład: ~27% klientów odchodzących

**Kluczowe cechy:**
- `tenure` - ile miesięcy klient jest w firmie
- `MonthlyCharges` - miesięczny rachunek
- `TotalCharges` - łączne płatności
- `Contract` - typ umowy (Month-to-month, One year, Two year)
- `InternetService` - rodzaj internetu
- `PaymentMethod` - metoda płatności

---

## 🔧 Metodologia

### 1. Przygotowanie danych
- Usunięcie kolumny `customerID`
- Konwersja `TotalCharges` na typ numeryczny
- Wypełnienie brakujących wartości zerem

### 2. Konfiguracja PyCaret
```python
setup(
    data=df,
    target='Churn',
    session_id=123,
    train_size=0.8,      # 80% trening, 20% test
    fold=5,              # 5-fold cross-validation
    normalize=True       # Normalizacja cech numerycznych
)
```

### 3. Wybór i trening modelu
- **Algorytm:** `compare_models(sort='Recall')` → wybór najlepszego pod kątem Recall
- **Model bazowy:** Gradient Boosting Classifier
- **Tuning:** `tune_model(optimize='Recall', n_iter=20)` - optymalizacja hiperparametrów

### 4. Analiza thresholdów
Testowane progi decyzyjne:
- **0.3** - Liberalny (więcej alarmów)
- **0.5** - Standardowy (domyślny)
- **0.7** - Konserwatywny (mało alarmów)

---

## 📈 Wyniki

### 🔧 Tuning vs Model Bazowy

| Metryka | Model Bazowy | Model po Tuningu | Zmiana |
|---------|--------------|------------------|--------|
| **Recall** | 0.8468 | 0.8481 | +0.13 p.p. |
| **Precision** | 0.6766 | 0.6779 | +0.13 p.p. |
| **Accuracy** | 0.8124 | 0.8131 | +0.07 p.p. |
| **AUC** | 0.8463 | 0.8471 | +0.08 p.p. |

**🎯 DECYZJA: TUNING NIE WART ZACHODU**

❌ Recall poprawił się tylko o **0.13 p.p.** (< 1.0 p.p. próg akceptacji)  
✅ **Używamy modelu bazowego** - prostszy, szybszy, równie dobry

**Wnioski:**
- Tuning nie dał znaczącej poprawy Recall
- Model bazowy ma wystarczającą jakość (Recall ~85%)
- Dodatkowa złożoność tuningu nie jest uzasadniona

---

### 🎯 Analiza Thresholdów

#### Confusion Matrix dla różnych thresholdów

| Threshold | TN | FP | FN | TP | Recall | Precision | Accuracy |
|-----------|----|----|----|----|--------|-----------|----------|
| **0.3** | 970 | 70 | 61 | 308 | **83.47%** | 81.48% | 90.70% |
| **0.5** | 970 | 70 | 61 | 308 | **83.47%** | 81.48% | 90.70% |
| **0.7** | 1017 | 23 | 116 | 253 | **68.56%** | 91.67% | 90.13% |

**⚠️ KLUCZOWA OBSERWACJA:**

Threshold **0.3 i 0.5 dają IDENTYCZNE wyniki**! Dlaczego?

Model ma **polaryzowany rozkład prawdopodobieństw**:
- `< 0.3`: **0 klientów**
- `0.3-0.5`: **0 klientów**
- `0.5-0.7`: **434 klientów**
- `≥ 0.7`: **975 klientów**

**Wszystkie przewidywania ≥ 0.5!** Model jest bardzo pewny swoich decyzji.

---

### 💰 Analiza Biznesowa

#### Założenia kosztowe:
- **Koszt fałszywego alarmu (FP):** 20 zł (telefon + czas konsultanta)
- **Koszt przegapienia klienta (FN):** 500 zł (utrata wartości rocznej)
- **Koszt próby retencji (TP):** 50 zł (telefon + oferta)
- **Skuteczność retencji:** 30% (zatrzymujemy 30% z TP)
- **Wartość zatrzymanego klienta:** 500 zł

#### Wyniki finansowe:

| Threshold | Koszty Łącznie | Przychody | **Zysk Netto** |
|-----------|----------------|-----------|----------------|
| **0.3** | 20,850 zł | 46,200 zł | **+25,350 zł** ✅ |
| **0.5** | 20,850 zł | 46,200 zł | **+25,350 zł** ✅ |
| **0.7** | 39,260 zł | 37,950 zł | **-1,310 zł** ❌ |

**Szczegółowa analiza threshold 0.7 (konserwatywny):**
- ❌ **116 przegapionych klientów** (FN) → koszt: 58,000 zł
- ✅ Tylko 23 fałszywe alarmy (FP) → oszczędność: 940 zł vs threshold 0.5
- **Bilans:** Oszczędność 940 zł na FP < Strata 27,500 zł na FN
- **Wynik:** -1,310 zł (strata!)

---

## 🎓 Kluczowe Wnioski

### 1. Tuning - Czy warto?

**❌ NIE w tym przypadku**

- Poprawa Recall: +0.13 p.p. (< 1% próg)
- Model bazowy ma już wysoki Recall (~85%)
- Tuning dodaje złożoność bez znaczącej korzyści
- **Rekomendacja:** Używaj modelu bazowego

### 2. Trade-off Recall vs Precision

| Threshold | Charakterystyka | Kiedy używać? |
|-----------|----------------|---------------|
| **0.3-0.4** | 🟢 Wysoki Recall<br>🔴 Niski Precision | Koszt przegapienia >> Koszt alarmu |
| **0.5** | 🟡 Balans | Standardowe podejście |
| **0.7+** | 🔴 Niski Recall<br>🟢 Wysoki Precision | Koszt alarmu >> Koszt przegapienia |

### 3. Dlaczego w churn zazwyczaj niższy threshold?

**Stosunek kosztów: 1:25**
- Fałszywy alarm: 20 zł
- Przegapienie: 500 zł

**Lepiej 10 niepotrzebnych telefonów niż stracić 1 klienta!**

**Analogia:** Wykrywacz dymu w domu
- Wolisz 10 fałszywych alarmów niż przegapić pożar
- Tak samo w churn: wolisz 10 niepotrzebnych telefonów niż stracić klienta

### 4. Polaryzowany rozkład prawdopodobieństw

**W naszym modelu:**
- Wszystkie przewidywania ≥ 0.5
- Threshold 0.3 = Threshold 0.5 (identyczne wyniki!)
- **Wniosek:** Zostaw threshold 0.5 (domyślny)

Zmiana thresholdu z 0.5 na 0.3 nie wpłynęła na wyniki, ponieważ model nie przypisuje żadnym klientom prawdopodobieństw churn poniżej 0.5. Dopiero podniesienie thresholdu do 0.7 spowodowało spadek recall, eliminując klientów o umiarkowanym ryzyku odejścia.

Threshold działa tylko tam, gdzie model jest „niepewny”

---

## 📊 Wizualizacje

### Confusion Matrix dla różnych thresholdów

```
THRESHOLD 0.5 (WYBRANE)               THRESHOLD 0.7
Recall: 83.47% | Precision: 81.48%   Recall: 68.56% | Precision: 91.67%

           Przewidywane                      Przewidywane
           No      Yes                       No      Yes
Real No   970      70                 Real No  1017     23
    Yes    61     308                     Yes  116    253
```

**Threshold 0.7:**
- ✅ Mniej fałszywych alarmów: 23 vs 70 (-67%)
- ❌ Więcej przegapionych: 116 vs 61 (+90%)
- 💸 Finansowo gorsze: -1,310 zł straty!

---

## 🚀 Rekomendacje Biznesowe

### ✅ Co wdrożyć:

1. **Model:** Gradient Boosting Classifier (bazowy, bez tuningu)
2. **Threshold:** 0.5 (domyślny) - w naszym przypadku identyczny z 0.3
3. **Strategia:** Priorytet dla Recall (wykrycie klientów odchodzących)

### 📞 Akcje retencyjne:

Dla klientów z prawdopodobieństwem odejścia ≥ 0.5:
- Telefon z działem retencji
- Oferta specjalna / rabat
- Analiza przyczyn niezadowolenia
- Follow-up po 2 tygodniach

### 📈 Oczekiwane wyniki:

**Miesięcznie (zakładając 1,409 klientów testowych):**
- Wykryte zagrożenia: **308 klientów**
- Fałszywe alarmy: **70 klientów** (koszty: 1,400 zł)
- Zatrzymani klienci: **~92 klientów** (30% z 308)
- **Zysk netto: +25,350 zł miesięcznie**

**Rocznie:**
- **Oszczędzone straty: ~304,200 zł**
- Koszt fałszywych alarmów: ~16,800 zł
- **ROI: ~1,700%**

---

## 💡 Najważniejsza Lekcja

**W churn prediction:**

🔴 **Koszt przegapienia > Koszt fałszywego alarmu**

Dlatego:
- ✅ Optymalizuj pod **Recall** (nie AUC!)
- ✅ Używaj **niższego threshold** (0.3-0.5)
- ✅ Toleruj fałszywe alarmy
- ❌ NIE optymalizuj pod Precision

**Pamiętaj:** Lepiej niepotrzebnie zadzwonić do 10 klientów, niż stracić 1 wartościowego klienta!

---

## 📁 Struktura projektu

```
06_churn_recall_threshold/
├── README.md                           # Ten plik
├── churn_recall_threshold.ipynb       # Notebook z pełną analizą
├── churn_recall_threshold.py          # Skrypt Python
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
└── models/                            # (modele zapisane przez PyCaret)
```

---

## 🛠️ Technologie

- **Python 3.8+**
- **PyCaret 3.x** - AutoML
- **Pandas** - manipulacja danymi
- **Scikit-learn** - metryki ML
- **Matplotlib / Seaborn** - wizualizacje

---

## 🎯 Następne kroki

Potencjalne rozszerzenia projektu:

1. **Kalibracja modelu** - Platt Scaling, Isotonic Regression
2. **Cost-Sensitive Learning** - wbudowanie kosztów w trening
3. **Analiza feature importance** - które cechy najbardziej wpływają na churn?
4. **Segmentacja klientów** - różne thresholdy dla różnych segmentów
5. **Temporal analysis** - jak zmieniają się przewidywania w czasie?
6. **A/B testing** - test threshold 0.5 vs 0.3 na produkcji

---

## 📚 Bibliografia i zasoby

- [PyCaret Documentation - Classification](https://pycaret.gitbook.io/docs/get-started/functions/classification)
- [Scikit-learn - Precision-Recall](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html)
- [IBM Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

**Autor:** Łukasz  
**Data:** Styczeń 2026  
**Projekt:** ML Portfolio - Churn Prediction Series
