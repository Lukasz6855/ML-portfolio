# 🤖 Machine Learning Portfolio

Portfolio projektów Machine Learning pokazujące praktyczne zastosowanie algorytmów ML w problemach biznesowych - od podstawowej klasyfikacji i regresji, przez clustering, aż po zaawansowane techniki optymalizacji modeli, deployment i interpretacji wyników.

## 📋 Spis treści

- [O portfolio](#-o-portfolio)
- [Technologie](#-technologie)
- [Przegląd projektów](#-przegląd-projektów)
- [Progresja umiejętności](#-progresja-umiejętności)
- [Jak korzystać z portfolio](#-jak-korzystać-z-portfolio)
- [Struktura projektów](#-struktura-projektów)

---

## 🎯 O portfolio

To portfolio dokumentuje moją ścieżkę nauki Machine Learning poprzez **8 praktycznych projektów**, które stopniowo wprowadzają coraz bardziej zaawansowane koncepcje i techniki. Każdy projekt zawiera:

- 📓 **Jupyter Notebook** z szczegółową analizą i komentarzami
- 🐍 **Skrypt Python** gotowy do użycia produkcyjnego
- 📊 **Wizualizacje** wyników i metryk
- 📝 **README** z opisem problemu, metodologii i wniosków
- 💾 **Zapisane modele** do dalszego wykorzystania

### Kluczowe aspekty portfolio:

✅ **Problemy biznesowe** - każdy projekt rozwiązuje rzeczywisty problem  
✅ **Best practices** - walidacja, unikanie overfittingu, właściwe metryki  
✅ **Analiza ROI** - ocena wartości biznesowej rozwiązań ML  
✅ **Production-ready** - separacja treningu od predykcji, metadata, deployment  
✅ **Explainable AI** - interpretacja modeli i wyjaśnianie predykcji

---

## 🛠 Technologie

**Core ML & Data Science:**
- **Python 3.8+**
- **PyCaret** - AutoML framework (klasyfikacja, regresja, clustering)
- **Scikit-learn** - algorytmy ML, metryki, preprocessing
- **Pandas** - manipulacja i analiza danych
- **NumPy** - obliczenia numeryczne

**Wizualizacje & Interpretacja:**
- **Matplotlib** - podstawowe wykresy
- **Seaborn** - zaawansowane wizualizacje statystyczne
- **SHAP** - interpretacja modeli ML (Shapley values)

**Deployment & Production:**
- **Pickle** - serializacja modeli
- **JSON** - metadata i konfiguracja

---

## 📚 Przegląd projektów

### 01. 🚢 Titanic - Klasyfikacja Binarna
**Problem:** Przewidywanie przeżycia pasażerów Titanica na podstawie danych demograficznych i informacji o podróży.

**Technika:** Binary Classification z AutoML (PyCaret)  
**Najlepszy model:** Logistic Regression  
**Wynik:** Accuracy **80.21%**

**Kluczowe umiejętności:**
- Pierwsza klasyfikacja binarna
- Setup PyCaret i compare_models
- Zapis i wczytanie modelu

📁 [Szczegóły projektu](01_titanic_classification/README.MD)

---

### 02. 🏠 House Prices - Regresja
**Problem:** Przewidywanie cen domów na podstawie ich parametrów (powierzchnia, lokalizacja, liczba pokoi itp.).

**Technika:** Regression z AutoML (PyCaret)  
**Najlepszy model:** Gradient Boosting Regressor  
**Wynik:** MAE **17,276** | RMSE **28,314**

**Kluczowe umiejętności:**
- Przejście od klasyfikacji do regresji
- Metryki regresyjne (MAE, RMSE, R²)
- Walidacja modeli regresyjnych

📁 [Szczegóły projektu](02_house_price_regression/README.md)

---

### 03. 🛍️ Customer Segmentation - Clustering
**Problem:** Segmentacja klientów centrum handlowego w celu personalizacji kampanii marketingowych.

**Technika:** K-Means Clustering z PyCaret  
**Wynik:** **5 segmentów klientów** z różnymi profileami zachowań

**Segmenty:**
- 💎 **VIP / Premium** - Wysokie dochody, wysokie wydatki (20%)
- 👥 **Przeciętni** - Średnie dochody i wydatki (23.5%)
- 🎯 **Młodzi entuzjaści** - Niskie dochody, wysokie wydatki (27%)
- 💰 **Oszczędni zamożni** - Wysokie dochody, niskie wydatki (14.5%)
- 🔵 **Budżetowi** - Niskie dochody i wydatki (15%)

**Kluczowe umiejętności:**
- Unsupervised learning
- Metoda Elbow do wyboru liczby klastrów
- Interpretacja biznesowa klastrów
- Wizualizacje 2D/3D

📁 [Szczegóły projektu](03_customer_segmentation_clustering/README.md)

---

### 04. 📱 Churn Prediction - Unikanie Overfittingu
**Problem:** Przewidywanie odejścia klientów firmy telekomunikacyjnej, aby zapobiegać churnowi poprzez kampanie retencyjne.

**Technika:** Classification z walidacją krzyżową (5-fold CV)  
**Najlepszy model:** Gradient Boosting Classifier  
**Wynik:** AUC **0.85** | Recall **75%**

**⚠️ Kluczowa lekcja: OVERFITTING**
- Demonstracja overfittingu (Decision Tree: Train 99%, Test 72%)
- Poprawna walidacja z cross-validation
- Training accuracy **NIE JEST** metryką sukcesu!

**Wartość biznesowa:**
- Identyfikacja 75% klientów zagrożonych odejściem
- Redukcja churnu o 20-30%
- ROI: Każdy utrzymany klient = 5x oszczędność kosztów akwizycji

**Kluczowe umiejętności:**
- Rozpoznawanie i unikanie overfittingu
- Cross-validation jako standard
- Business value analysis

📁 [Szczegóły projektu](04_churn_overfitting/README.md)

---

### 05. 🎯 Churn - Tuning Modelu (Czy warto?)
**Problem:** Analiza ROI hyperparameter tuningu - czy optymalizacja parametrów modelu daje realną wartość biznesową?

**Eksperyment:** Model bazowy vs 20 iteracji tuningu  
**Wynik tuningu:** AUC +0.30 p.p., ale Accuracy/Recall/Precision **spadły**

**💰 Analiza biznesowa:**
- Wykryci klienci: **-14** po tuningu
- Zatrzymani klienci: **-4** po tuningu
- **Strata finansowa:** -2,000 zł rocznie
- **Czas tuningu:** 15+ minut

**🎯 Wniosek:** **TUNING NIE WART ZACHODU**

**Kluczowe umiejętności:**
- Krytyczna ocena technik ML
- Analiza ROI i kosztu czasu
- Decyzje oparte na business value, nie tylko metrykach

📁 [Szczegóły projektu](05_churn_model_tuning/README.md)

---

### 06. ⚖️ Churn - Threshold i Trade-off Recall/Precision
**Problem:** Optymalizacja progu decyzyjnego (threshold) dla maksymalizacji wykrywalności klientów odchodzących przy akceptowalnym poziomie fałszywych alarmów.

**Analiza:** 3 thresholdy (0.3, 0.5, 0.7)

**Wyniki biznesowe:**

| Threshold | Recall | Wykryci | Akcje retencyjne | Koszt | Przychód | **Zysk** |
|-----------|--------|---------|------------------|-------|----------|----------|
| **0.3** (liberalny) | **95%** | **2,565** | 2,565 | 128k | 230k | **+102k** ✅ |
| **0.5** (standard) | 84% | 2,268 | 2,268 | 113k | 204k | **+91k** |
| **0.7** (konserwatywny) | 62% | 1,674 | 1,674 | 84k | 150k | **+66k** |

**🎯 Najlepszy wybór:** Threshold **0.3** (+102k zysku rocznie)

**Kluczowe umiejętności:**
- Rozumienie trade-off Recall vs Precision
- Dostosowanie threshold do celów biznesowych
- W churn prediction: lepiej więcej false positives niż false negatives

📁 [Szczegóły projektu](06_churn_recall_threshold/README.md)

---

### 07. 🚀 Churn - Model Deployment
**Problem:** Production-ready deployment modelu churn prediction z pełnym cyklem trenowania i predykcji.

**Architektura:**
1. **train.ipynb/train.py** - Trenowanie i zapis modelu
2. **predict.ipynb/predict.py** - Wczytanie modelu i predykcje
3. **metadata.json** - Konfiguracja (threshold, optimization settings)
4. **predictions_summary.csv** - Wyniki z poziomami ryzyka

**Poziomy ryzyka:**
- 🔴 **HIGH** (prob > 0.7) - Natychmiastowy kontakt
- 🟡 **MEDIUM** (0.5-0.7) - Monitoring i proaktywne oferty
- 🟢 **LOW** (< 0.5) - Standardowa obsługa

**Kluczowe umiejętności:**
- Separacja train/predict pipeline
- Obsługa prediction_score w PyCaret
- Metadata jako konfiguracja
- Production-ready CSV outputs dla systemów biznesowych

📁 [Szczegóły projektu](07_churn_model_deployment/README.md)

---

### 08. 🔍 Churn - Explainable AI (SHAP)
**Problem:** **Nie wystarczy wiedzieć, że klient odejdzie - musimy wiedzieć DLACZEGO!**

**Techniki interpretacji:**
1. **Feature Importance** - ranking cech z modelu
2. **SHAP Summary Plot** - kierunek i siła wpływu każdej cechy
3. **SHAP Bar Plot** - proste zestawienie dla managementu
4. **SHAP Force/Waterfall Plot** - analiza pojedynczego klienta

**🔴 TOP 5 Czynników Odejść:**

1. **Tenure (< 6 miesięcy)** ⭐ NAJWAŻNIEJSZY
   - Nowi klienci mają DRASTYCZNIE wyższe ryzyko
   - Akcja: Welcome program, częsty kontakt, rabaty

2. **MonthlyCharges (wysokie)**
   - Wysoka cena irytuje klientów
   - Akcja: Więcej value za tę samą cenę, targetowane rabaty

3. **TotalCharges (niskie)**
   - Niskie TotalCharges = krótki staż = brak lojalności
   - Akcja: Programy lojalnościowe dla długoterminowych relacji

4. **Contract (Month-to-month)**
   - Brak zobowiązania = łatwe odejście
   - Akcja: Zachęty do rocznych/2-letnich kontraktów (rabaty 15-25%)

5. **Fiber optic + Electronic check**
   - Wysokie oczekiwania + niewygodna płatność
   - Akcja: Automatyczne płatności, edukacja, special care

**Kluczowe umiejętności:**
- Interpretacja czarnej skrzynki ML
- SHAP values i Shapley theory
- Tłumaczenie wyników technicznych na język biznesowy
- Konkretne rekomendacje akcji retencyjnych

📁 [Szczegóły projektu](08_churn_model_explainability/README.md)

---

## 📈 Progresja umiejętności

Portfolio pokazuje naturalną progresję od podstaw do zaawansowanych technik:

### Poziom 1: Podstawy ML (Projekty 01-03)
- ✅ Klasyfikacja binarna (Titanic)
- ✅ Regresja (House Prices)
- ✅ Clustering (Customer Segmentation)
- ✅ AutoML z PyCaret
- ✅ Podstawowe metryki

### Poziom 2: Best Practices (Projekt 04)
- ✅ Overfitting i jak go unikać
- ✅ Cross-validation
- ✅ Poprawna walidacja modeli
- ✅ Business value analysis

### Poziom 3: Optymalizacja (Projekty 05-06)
- ✅ Hyperparameter tuning + analiza ROI
- ✅ Threshold optimization
- ✅ Trade-off Recall/Precision
- ✅ Decyzje biznesowe oparte na danych

### Poziom 4: Production & Explainability (Projekty 07-08)
- ✅ Production-ready deployment
- ✅ Train/predict separation
- ✅ Metadata i konfiguracja
- ✅ Explainable AI (SHAP)
- ✅ Interpretacja dla biznesu

---

## 🚀 Jak korzystać z portfolio

### Dla rekruterów i pracodawców:

1. **Quick overview:** Zobacz [Przegląd projektów](#-przegląd-projektów) dla szybkiego zrozumienia zakresu
2. **Głębsza analiza:** Każdy projekt ma README z business case, metodyką i wnioskami
3. **Kod:** Notebooki z komentarzami + skrypty Python gotowe do produkcji
4. **Progresja:** Portfolio pokazuje systematyczną naukę od podstaw do zaawansowanych technik

### Dla uczących się ML:

1. **Zacznij od projektu 01** - stopniuj trudność
2. **Uruchom notebooki** - każdy zawiera szczegółowe komentarze
3. **Zwróć uwagę na best practices:**
   - Projekt 04: Jak unikać overfittingu
   - Projekt 05: Krytyczna ocena technik (tuning)
   - Projekt 06: Optymalizacja dla biznesu
   - Projekt 08: Interpretacja modeli

### Instalacja zależności:

```bash
pip install pycaret pandas numpy matplotlib seaborn scikit-learn shap
```

---

## 📂 Struktura projektów

Każdy projekt ma spójną strukturę:

```
XX_project_name/
│
├── README.md                    # Szczegółowy opis projektu
├── notebook.ipynb               # Jupyter Notebook z analizą
├── script.py                    # Skrypt Python (produkcja)
│
├── data/                        # Dane wejściowe i wyjściowe
│   ├── original_dataset.csv
│   └── results.csv
│
└── models/                      # Zapisane modele
    ├── model.pkl
    └── metadata.json
```

---

## 📊 Podsumowanie wyników

| Projekt | Problem | Model | Metryka | Wartość |
|---------|---------|-------|---------|---------|
| 01 Titanic | Klasyfikacja | Logistic Regression | Accuracy | **80.21%** |
| 02 Houses | Regresja | Gradient Boosting | MAE / RMSE | **17.3k / 28.3k** |
| 03 Customers | Clustering | K-Means | Liczba klastrów | **5 segmentów** |
| 04 Churn | Klasyfikacja | Gradient Boosting | AUC / Recall | **0.85 / 75%** |
| 05 Churn Tuning | Optymalizacja | - | ROI tuningu | **-2,000 zł** ❌ |
| 06 Churn Threshold | Optymalizacja | - | Najlepszy threshold | **0.3 (+102k)** ✅ |
| 07 Churn Deploy | Deployment | Production-ready | - | **3 poziomy ryzyka** |
| 08 Churn Explain | Interpretacja | SHAP | Top feature | **Tenure** ⭐ |

---

## 🎓 Wnioski z portfolio

### Techniczne:
✅ AutoML (PyCaret) przyspiesza development, ale wymaga technicznej krytycznej oceny  
✅ Cross-validation jest absolutnym standardem - training accuracy jest bez wartości  
✅ Tuning nie zawsze daje wartość - liczy się business impact, nie perfekcyjne metryki  
✅ Threshold optimization może być ważniejszy niż wybór algorytmu  
✅ Explainable AI to must-have dla produkcyjnych modeli

### Biznesowe:
💰 Model ML bez biznesowej analizy ROI to zabawka, nie rozwiązanie  
💰 Koszt False Negative vs False Positive determinuje strategię optymalizacji  
💰 Interpretacja modelu jest kluczowa dla akcji i zaufania stakeholderów  
💰 Production-ready = separacja train/predict + metadata + monitoring

---

## 📧 Kontakt

Jeśli masz pytania o portfolio lub chcesz omówić współpracę:

- **GitHub:** [https://github.com/Lukasz6855]
- **LinkedIn:** [https://www.linkedin.com/in/lukasz-s-01754b3ab/]
- **Email:** [lukasz6855@gmail.com]

---

## 📜 Licencja

Projekty edukacyjne - wolne do użytku z podaniem źródła.

**Datasets:**
- Titanic: [Kaggle - Titanic Dataset](https://www.kaggle.com/c/titanic)
- House Prices: [Kaggle - House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- Mall Customers: [Kaggle - Mall Customers](https://www.kaggle.com/datasets)
- Telco Churn: [IBM Watson Analytics](https://www.kaggle.com/datasets)

---

**Data ostatniej aktualizacji:** 20.02.2026

**Status:** ✅ Portfolio kompletne (8/8 projektów)
