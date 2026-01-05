# 🎯 Tuning Modelu - Czy Warto?

## 📋 Spis treści
- [Cel projektu](#-cel-projektu)
- [Technologie](#-technologie)
- [Kluczowe pytanie](#-kluczowe-pytanie-tuning--czy-warto)
- [Wyniki eksperymentu](#-wyniki-eksperymentu)
- [Analiza biznesowa](#-analiza-biznesowa)
- [Wnioski](#-wnioski)
- [Jak uruchomić](#-jak-uruchomić)

## 🎯 Cel projektu

Projekt odpowiada na **fundamentalne pytanie** w Machine Learning:

> **Czy tuning modelu (optymalizacja hiperparametrów) daje realną wartość biznesową?**

Wiele osób wykonuje tuning "bo tak trzeba", nie zastanawiając się czy:
- ✅ Poprawa techniczna jest istotna
- ✅ Przekłada się na korzyści biznesowe
- ✅ Czas poświęcony na tuning jest uzasadniony

Ten projekt przeprowadza **rzetelną analizę techniczno-biznesową** na rzeczywistych danych.

## 🛠 Technologie

- **Python 3.8+**
- **PyCaret 3.x** - AutoML do klasyfikacji
- **Pandas** - manipulacja danymi
- **NumPy** - obliczenia numeryczne

### Dataset
**Telco Customer Churn** - dane o klientach firmy telekomunikacyjnej:
- 7043 klientów
- 20 cech (tenure, contract, monthly charges, itp.)
- Target: Churn (Yes/No) - 27% klientów odchodzi

## ❓ Kluczowe pytanie: Tuning – Czy warto?

### Metodologia

1. **Model bazowy** - Gradient Boosting z domyślnymi ustawieniami
2. **Model po tuningu** - 20 iteracji optymalizacji z Optuna
3. **Porównanie metryk** - Accuracy, AUC, Recall, Precision
4. **Analiza biznesowa** - Obliczenie ROI i zysku/straty

### Co testujemy?

```python
# Model PRZED tuningiem (domyślne ustawienia)
model_before = create_model('gbc', fold=5)

# Model PO tuningu (optymalizacja 20 konfiguracji)
model_after = tune_model(model_before, optimize='AUC', n_iter=20)
```

## 📊 Wyniki eksperymentu

### Metryki techniczne

| Metryka | Przed tuningiem | Po tuningu | Zmiana |
|---------|----------------|------------|--------|
| **Accuracy** | 79.93% | 79.41% | **-0.52 p.p.** ⚠️ |
| **AUC** | 0.8463 | 0.8493 | **+0.30 p.p.** ✅ |
| **Recall** | 79.93% | 79.41% | **-0.52 p.p.** ⚠️ |
| **Precision** | 79.03% | 78.21% | **-0.82 p.p.** ⚠️ |

### 🔍 Interpretacja metryk

**AUC (0.8463 → 0.8493):**
- ✅ Nieznaczna poprawa o 0.3 punktu procentowego
- Model nieco lepiej rozróżnia klientów odchodzących vs zostających
- **Ale:** Poprawa jest minimalna

**Accuracy, Recall, Precision:**
- ❌ **Wszystkie spadły** po tuningu!
- Model po tuningu wykrywa **MNIEJ** klientów zagrożonych odejściem
- Precision spadła - więcej fałszywych alarmów

## 💰 Analiza biznesowa

### Założenia

```python
Baza klientów: 10,000
Klienci odchodzący rocznie: 2,700 (27%)
Koszt próby zatrzymania klienta: 50 zł (telefon + oferta)
Wartość klienta rocznie: 500 zł
Skuteczność retencji: 30% (zatrzymujemy 30% wykrytych)
```

### Wykrywanie klientów

| | Przed tuningiem | Po tuningu | Różnica |
|---|----------------|-----------|---------|
| **Wykryci klienci** | 2,158 | 2,144 | **-14** ❌ |
| **Zatrzymani (30%)** | 647 | 643 | **-4** ❌ |

### Bilans finansowy

```
📉 STRATA Z TUNINGU:

Dodatkowy koszt:     -700 zł  (14 klientów mniej × 50 zł)
Dodatkowy przychód:  -2,000 zł (4 klientów mniej × 500 zł)
─────────────────────────────
STRATA NETTO:        -1,300 zł rocznie
```

### 📈 ROI (Return on Investment)

```
ROI = -185% (strata!)

Inwestujemy czas w tuning → tracę 1,300 zł rocznie
```

## 🎯 Wnioski

### 1. Czy tuning poprawił model technicznie?

**⚠️ Minimalna poprawa, praktycznie bez znaczenia**

- AUC wzrosło o 0.30 p.p. - nieistotne statystycznie
- Accuracy, Recall, Precision **spadły**
- Model po tuningu jest **gorszy** w wykrywaniu klientów

### 2. Czy tuning ma sens biznesowy?

**❌ NIE - Model po tuningu GORZEJ wykrywa klientów i generuje STRATĘ**

- Wykrywamy **14 klientów MNIEJ** rocznie
- Zatrzymujemy **4 klientów MNIEJ**
- **Strata: 1,300 zł rocznie**

### 3. Kiedy tuning MA SENS?

Tuning jest wart czasu i wysiłku gdy:

✅ **Poprawa AUC > 1 punkt procentowy** (u nas: 0.30 p.p.)
✅ **Recall/Precision rosną** (u nas: spadły!)
✅ **Zysk netto > 0** (u nas: -1,300 zł)
✅ **Czas tuningu < wartość poprawy** (u nas: nie dotyczy)

### 4. **OSTATECZNA REKOMENDACJA**

```
🛑 ZOSTAŃ PRZY MODELU PODSTAWOWYM

Powody:
- Model bazowy (bez tuningu) jest lepszy
- Tuning nie tylko nie pomógł, ale pogorszył wyniki
- Model bazowy: Recall 79.93%, Precision 79.03%
- Model tuned: Recall 79.41%, Precision 78.21%
```

## 💡 Kluczowe lekcje

### Dla Data Scientists:

1. **Tuning ≠ Automatyczna poprawa** - czasem może pogorszyć model
2. **Zawsze porównuj PRZED vs PO** - nigdy nie zakładaj, że tuning pomoże
3. **Cross-validation jest kluczowe** - chroni przed overfittingiem
4. **Dobry baseline to podstawa** - model z domyślnymi ustawieniami może być wystarczający

### Dla biznesu:

1. **Nie każda "optymalizacja" się opłaca** - czas ma wartość
2. **Model prostszy może być lepszy** - mniej ryzyka, łatwiejsze utrzymanie
3. **Metryki techniczne ≠ Wartość biznesowa** - zawsze obliczaj ROI
4. **80% jakości w 20% czasu** - często wystarcza model bazowy

## 📈 Porównanie z innymi projektami

W naszym portfolio mamy 3 modele churn:

| Projekt | Model | AUC | Recall | Czy tuning? | Wynik |
|---------|-------|-----|--------|-------------|-------|
| 04_churn_overfitting | Gradient Boosting | 0.8463 | 79.93% | ❌ Nie | ✅ Świetny! |
| **05_churn_tuning** | **GB (tuned)** | **0.8493** | **79.41%** | **✅ Tak** | **⚠️ Gorszy!** |

**Wniosek:** Model z projektu 04 (bez tuningu) jest **LEPSZY** niż model z tuningu!

## 🚀 Jak uruchomić

### Instalacja

```bash
# Sklonuj repozytorium
git clone https://github.com/your-username/ML-portfolio.git
cd ML-portfolio/05_churn_model_tuning

# Utwórz środowisko wirtualne
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Zainstaluj zależności
pip install pandas numpy pycaret scikit-learn
```

### Uruchomienie

**Notebook (zalecane):**
```bash
jupyter notebook churn_tuning.ipynb
```

**Skrypt Python:**
```bash
python churn_tuning.py
```

### Struktura projektu

```
05_churn_model_tuning/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── models/
│   └── churn_tuned_model.pkl  (zapisany model po tuningu)
│
├── churn_tuning.ipynb  (notebook z pełnymi wyjaśnieniami)
├── churn_tuning.py     (skrypt Python)
└── README.md           (ten plik)
```

## 📚 Dodatkowe materiały

### Co to jest tuning?

Tuning (optymalizacja hiperparametrów) to proces szukania **najlepszych ustawień** algorytmu ML:

```python
# Przykładowe hiperparametry dla Gradient Boosting:
- learning_rate: jak szybko model uczy się
- n_estimators: ile drzew decyzyjnych
- max_depth: głębokość drzew
- min_samples_split: minimalna liczba próbek do podziału
```

### Metody tuningu

1. **Grid Search** - testuje wszystkie kombinacje (wolne)
2. **Random Search** - losowe kombinacje (szybsze)
3. **Optuna** - inteligentne przeszukiwanie (użyte w tym projekcie)

### Metryki wyjaśnione

- **Accuracy** = (TP + TN) / All - ogólna dokładność
- **AUC** = pole pod krzywą ROC - zdolność rozróżniania
- **Recall** = TP / (TP + FN) - ile % odchodzących wykrywamy
- **Precision** = TP / (TP + FP) - ile % alertów jest trafnych

## 🎓 Wnioski końcowe

### Główna konkluzja:

> **Tuning nie zawsze poprawia model. W tym przypadku model bazowy (bez tuningu) okazał się LEPSZY.**

### Praktyczne rekomendacje:

1. ✅ **Zawsze trenuj model bazowy** - często jest wystarczający
2. ✅ **Porównuj wyniki obiektywnie** - nie zakładaj, że tuning pomoże
3. ✅ **Obliczaj ROI** - czas to pieniądz
4. ✅ **Prostota > Złożoność** - prostszy model łatwiejszy w utrzymaniu

### Co dalej?

Jeśli szukasz **DZIAŁAJĄCEGO** modelu churn, sprawdź:
- **Projekt 04: churn_overfitting** - model bazowy z AUC 0.8463 i Recall 79.93%

Ten projekt pokazał, że **nie wszystkie techniki ML zawsze działają** - to cenna lekcja! 🎯

---

**Autor:** Łukasz  
**Data:** Styczeń 2026
