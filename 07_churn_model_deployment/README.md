# 🚀 Projekt 7: Deployment-Ready Churn Prediction Model

## 📋 Opis projektu

Projekt demonstruje **produkcyjne wdrożenie modelu churn prediction** z pełnym cyklem:
1. **Trenowanie modelu** (train.ipynb) - selekcja najlepszego algorytmu z optymalizacją na Recall
2. **Zapisanie modelu** - serializacja do .pkl + metadata.json z konfiguracją
3. **Predykcje dla nowych klientów** (predict.ipynb) - załadowanie modelu i wykonanie predykcji

**Kluczowe aspekty produkcyjne:**
- ✅ Separacja treningu od predykcji
- ✅ Metadata.json jako konfiguracja (threshold, optimization settings)
- ✅ Obsługa prediction_score (prawdopodobieństwo dla predicted class)
- ✅ Poziomy ryzyka (HIGH/MEDIUM/LOW) dla akcji retencyjnych
- ✅ Gotowe pliki CSV dla systemów biznesowych

---

## 🎯 Cel projektu

**Pokazać kompletny workflow produkcyjny:**
- Jak zapisać model z ustawieniami biznesowymi
- Jak wczytać model i zastosować threshold z konfiguracji
- Jak poprawnie interpretować prediction_score w PyCaret
- Jak wygenerować rekomendacje akcji dla biznesu

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

## 🔧 Struktura projektu

```
07_churn_model_deployment/
├── train.ipynb              # Trenowanie i zapisanie modelu
├── predict.ipynb            # Predykcje dla nowych klientów
├── train.py                 # Uproszczona wersja skryptowa (trenowanie)
├── predict.py               # Uproszczona wersja skryptowa (predykcje)
├── README.md                # Dokumentacja projektu
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Oryginalny dataset
│   ├── new_customers.csv                      # Nowi klienci do oceny
│   ├── predictions_results.csv                # Pełne wyniki predykcji
│   └── predictions_summary.csv                # Podsumowanie z rekomendacjami
└── models/
    ├── churn_model.pkl      # Zapisany model + preprocessing pipeline
    └── metadata.json        # Konfiguracja (threshold, optimization)
```

---

## 📚 Workflow

### **Krok 1: Trenowanie modelu** (train.ipynb)

```python
# 1. Wczytanie danych
df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# 2. Czyszczenie
df = df.drop('customerID', axis=1)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

# 3. Setup PyCaret
s = setup(data=df, target='Churn', 
          train_size=0.8, 
          fold=5,
          normalize=True,
          session_id=123)

# 4. Porównanie modeli (sortowane po Recall)
best_model = compare_models(sort='Recall')

# 5. Zapis modelu
save_model(best_model, 'models/churn_model')

# 6. Zapis metadanych
metadata = {
    "threshold": 0.5,
    "optimized_for": "recall",
    "business_reason": "false negatives are costly",
    "model_type": type(best_model).__name__,
    "train_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}
with open('models/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
```

**Wyjście:**
- `models/churn_model.pkl` - model + preprocessing pipeline
- `models/metadata.json` - ustawienia biznesowe

---

### **Krok 2: Predykcje dla nowych klientów** (predict.ipynb)

```python
# 1. Przygotowanie nowych klientów (symulacja)
df_original = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
new_customers = df_original.sample(20, random_state=123)
new_customers = new_customers.drop('Churn', axis=1)  # Usunięcie target
new_customers.to_csv('data/new_customers.csv', index=False)

# 2. Wczytanie modelu i metadanych
model = load_model('models/churn_model')

with open('models/metadata.json', 'r') as f:
    metadata = json.load(f)
threshold = metadata['threshold']

# 3. Wczytanie klientów
customers = pd.read_csv('data/new_customers.csv')

# 4. WAŻNE: Predykcje z threshold z metadata.json!
predictions = predict_model(model, data=customers, 
                           probability_threshold=threshold)

# 5. Analiza wyników
results = predictions[['customerID', 'tenure', 'MonthlyCharges', 
                       'prediction_label', 'prediction_score']].copy()

# 6. Poziomy ryzyka - TYLKO dla prediction_label = Yes
def get_risk_level(prob):
    if prob >= 0.7: return "HIGH"
    elif prob >= 0.5: return "MEDIUM"
    else: return "LOW"

results['risk_level'] = results.apply(
    lambda row: get_risk_level(row['prediction_score']) 
                if row['prediction_label'] == 'Yes' 
                else 'LOW',
    axis=1
)

# 7. Zapis wyników
predictions.to_csv('data/predictions_results.csv', index=False)
results.to_csv('data/predictions_summary.csv', index=False)
```

**Wyjście:**
- `data/predictions_results.csv` - pełne dane + predykcje
- `data/predictions_summary.csv` - podsumowanie + poziomy ryzyka

---

## 🔑 Kluczowe koncepty

### **1. Metadata.json - konfiguracja bez zmiany kodu**

```json
{
  "threshold": 0.5,
  "optimized_for": "recall",
  "business_reason": "false negatives are costly",
  "model_type": "LogisticRegression",
  "train_date": "2026-01-07 20:26:39"
}
```

**Zalety:**
- ✅ Business user może zmienić threshold bez programisty
- ✅ Historia zmian w Git
- ✅ Jeden punkt konfiguracji
- ✅ Dokumentacja decyzji biznesowych

**Użycie:**
```python
# Zamiast hardkodowania:
predictions = predict_model(model, data, probability_threshold=0.5)

# Używamy z metadata:
threshold = metadata['threshold']
predictions = predict_model(model, data, probability_threshold=threshold)
```

Teraz zmiana threshold to edycja JSON → rerun skryptu!

---

### **2. prediction_score w PyCaret**

**⚠️ UWAGA:** `prediction_score` NIE jest zawsze prawdopodobieństwem odejścia!

```
prediction_score = prawdopodobieństwo dla predicted class
```

**Przykłady:**

| prediction_label | prediction_score | Znaczenie |
|-----------------|------------------|-----------|
| Yes | 0.85 | 85% pewności że **ODEJDZIE** |
| No | 0.73 | 73% pewności że **ZOSTANIE** |

**Dlaczego to ważne?**

❌ **Błędne podejście:**
```python
# Błąd: stosowanie threshold do prediction_score
results['churn'] = results['prediction_score'].apply(
    lambda x: 'Yes' if x >= 0.5 else 'No'
)
```
To zakłada że score = prawdopodobieństwo dla "Yes", ale nie zawsze!

✅ **Prawidłowe podejście:**
```python
# 1. Użyj prediction_label (PyCaret już zastosował threshold)
churn_yes = results[results['prediction_label'] == 'Yes']

# 2. Dla poziomów ryzyka - TYLKO dla prediction_label = Yes
results['risk_level'] = results.apply(
    lambda row: get_risk_level(row['prediction_score']) 
                if row['prediction_label'] == 'Yes' 
                else 'LOW',
    axis=1
)
```

---

### **3. Poziomy ryzyka dla akcji retencyjnych**

**Kategorie:**
- 🔴 **HIGH RISK** (score ≥ 0.7) - 70%+ pewności odejścia
  - Akcja: PILNY kontakt z działem retencji + oferta specjalna
  
- 🟡 **MEDIUM RISK** (score 0.5-0.7) - 50-70% pewności
  - Akcja: Kontakt telefoniczny + analiza przyczyn
  
- 🟢 **LOW RISK** (prediction_label = No)
  - Akcja: Monitoring standardowy

**Implementacja:**
```python
def get_risk_level(prob):
    """Dla klientów z prediction_label = Yes"""
    if prob >= 0.7: return "HIGH"
    elif prob >= 0.5: return "MEDIUM"
    else: return "LOW"

# Stosujemy TYLKO dla Yes (dla nich score = prawdopodobieństwo odejścia)
results['risk_level'] = results.apply(
    lambda row: get_risk_level(row['prediction_score']) 
                if row['prediction_label'] == 'Yes' 
                else 'LOW',
    axis=1
)
```

---

## 📈 Wyniki

### **Przykładowe predykcje (20 klientów):**

```
================================================================================
📊 PODSUMOWANIE
================================================================================

👥 Liczba klientów: 20
🔴 Przewidywane ODEJŚCIA (Churn = Yes): 3 (15.0%)
🟢 Przewidywane POZOSTANIE (Churn = No): 17 (85.0%)

📈 Statystyki pewności dla klientów z ryzykiem odejścia:
   Średnia pewność: 0.6234
   Minimum: 0.5012
   Maximum: 0.7891

================================================================================
🎯 REKOMENDACJE AKCJI
================================================================================

⚠️ KLIENCI WYMAGAJĄCY UWAGI: 3

👤 Klient: 5678-ABCDE
   Prawdopodobieństwo: 78.91%
   Poziom ryzyka: 🔴 HIGH RISK
   Akcja: PILNE: Natychmiastowy kontakt z działem retencji + oferta specjalna

👤 Klient: 1234-FGHIJ
   Prawdopodobieństwo: 62.15%
   Poziom ryzyka: 🟡 MEDIUM RISK
   Akcja: Kontakt telefoniczny + analiza przyczyn niezadowolenia

👤 Klient: 9012-KLMNO
   Prawdopodobieństwo: 50.12%
   Poziom ryzyka: 🟡 MEDIUM RISK
   Akcja: Kontakt telefoniczny + analiza przyczyn niezadowolenia
```

---

## 💡 Wnioski i best practices

### **1. Separacja treningu od predykcji**
- ✅ Model trenowany raz, używany wielokrotnie
- ✅ Predykcje mogą być uruchamiane codziennie/co tydzień
- ✅ Różne osoby: Data Scientist (trening) vs Business User (predykcje)

### **2. Metadata.json jako konfiguracja**
- ✅ Łatwa zmiana threshold bez edycji kodu
- ✅ Historia zmian w Git
- ✅ Dokumentacja decyzji biznesowych

### **3. Zrozumienie prediction_score**
- ⚠️ To NIE jest zawsze prawdopodobieństwo dla "Yes"!
- ✅ To prawdopodobieństwo dla predicted class
- ✅ Używaj prediction_label dla decyzji
- ✅ Używaj prediction_score dla poziomów ryzyka TYLKO gdy label = Yes

### **4. Poziomy ryzyka dla biznesu**
- ✅ HIGH/MEDIUM/LOW zamiast suchych liczb
- ✅ Konkretne akcje dla każdego poziomu
- ✅ Priorytetyzacja klientów do kontaktu

---

## 🚀 Jak uruchomić projekt?

### **1. Trenowanie modelu:**
```bash
# Jupyter Notebook:
jupyter notebook train.ipynb

# Lub skrypt:
python train.py
```

**Wyjście:**
- `models/churn_model.pkl`
- `models/metadata.json`

---

### **2. Predykcje dla nowych klientów:**
```bash
# Jupyter Notebook:
jupyter notebook predict.ipynb

# Lub skrypt:
python predict.py
```

**Wyjście:**
- `data/predictions_results.csv` (pełne dane)
- `data/predictions_summary.csv` (podsumowanie + rekomendacje)

---

### **3. Zmiana threshold (eksperyment):**

1. Otwórz `models/metadata.json`
2. Zmień `"threshold": 0.5` na `"threshold": 0.3`
3. Zapisz plik
4. Uruchom ponownie `predict.ipynb` lub `predict.py`

**Efekt:** Więcej klientów zostanie oznaczonych jako Churn = Yes (bardziej ostrożne podejście)

---

## 🔗 Powiązane projekty

- **Projekt 04**: Podstawowa analiza churn + handling overfitting
- **Projekt 05**: Feature engineering + business interpretation
- **Projekt 06**: Threshold analysis (0.3 vs 0.5 vs 0.7) + tuning comparison

**Projekt 07** to kulminacja - gotowy do produkcji system z pełnym workflow!

---

## 📦 Wymagania

```bash
pip install pandas
pip install pycaret
pip install scikit-learn
```

**Wersje:**
- Python 3.8+
- PyCaret 3.0+
- Pandas 1.3+

---

## 📞 Dla kogo jest ten projekt?

- ✅ **Data Scientists** - workflow trenowanie → deployment
- ✅ **ML Engineers** - serializacja modeli, metadata, konfiguracja
- ✅ **Business Analysts** - interpretacja wyników, poziomy ryzyka
- ✅ **Software Engineers** - integracja ML z systemami produkcyjnymi

---

## 🎓 Czego się nauczysz?

1. ✅ Jak zapisać i wczytać model PyCaret
2. ✅ Jak używać metadata.json dla konfiguracji biznesowej
3. ✅ Jak poprawnie interpretować prediction_score
4. ✅ Jak wygenerować rekomendacje akcji dla biznesu
5. ✅ Jak zbudować deployment-ready ML system

---

## 📄 Licencja

Dataset: IBM Watson Analytics / Kaggle
Kod: Do użytku edukacyjnego

---

**🚀 Model gotowy do produkcji!**
