# 🔍 Projekt 8: Interpretacja Modelu Churn - Dlaczego Klienci Odchodzą?

## 🎯 Cel projektu

**Główne założenie:** Interpretacja wyników modelu machine learning, które wpływają na predykcję odejścia klientów (churn).

**Nie wystarczy wiedzieć, że klient odejdzie - musimy wiedzieć DLACZEGO!**

Ten projekt pokazuje, jak używać zaawansowanych technik interpretacji modeli ML (Feature Importance i SHAP) do zrozumienia przyczyn odejść klientów i podejmowania konkretnych działań biznesowych.

## 📚 Czego się nauczysz

- ✅ Wczytywanie i analiza zapisanych modeli PyCaret
- ✅ Feature Importance - identyfikacja najważniejszych cech
- ✅ SHAP (SHapley Additive exPlanations) - głęboka interpretacja modelu
- ✅ Tworzenie wizualizacji: Summary Plot, Bar Plot, Force Plot
- ✅ Tłumaczenie wyników technicznych na język biznesowy
- ✅ Formułowanie konkretnych działań retencyjnych

## 🛠️ Technologie

- **Python 3.8+**
- **PyCaret** - framework do ML (wczytywanie modeli)
- **SHAP** - interpretacja modeli ML
- **Pandas** - analiza danych
- **Matplotlib/Seaborn** - wizualizacje
- **NumPy** - operacje numeryczne

## 📂 Struktura projektu

```
08_churn_model_explainability/
│
├── churn_explainability.ipynb    # Notebook z pełnymi objaśnieniami
├── churn_explainability.py       # Skrypt Python z komentarzami
├── README.md                     # Ten plik
│
├── model/                        # Folder z zapisanym modelem
│   ├── churn_model.pkl          # Wytrenowany model PyCaret (Logistic Regression)
│   └── metadata.json            # Metadane modelu
│
├── data/                         # Wygenerowane dane
│   └── feature_importance_shap.csv
│
└── plots/                        # Wygenerowane wizualizacje
    ├── Feature Importance.png
    ├── shap_summary_plot.png
    ├── shap_bar_plot.png
    ├── shap_force_plot_customer_0.png
    └── shap_waterfall_plot_customer_0.png
```

## 🚀 Jak uruchomić

### 1. Instalacja zależności

```bash
pip install pycaret shap pandas matplotlib seaborn numpy
```

### 2. Uruchomienie notebooka

```bash
jupyter notebook churn_explainability.ipynb
```

### 3. Uruchomienie skryptu Python

```bash
python churn_explainability.py
```

## 📊 Czego dowiesz się z analizy

### 1. **Feature Importance (Logistic Regression)**
- Ranking najważniejszych cech z najlepszego modelu produkcyjnego
- Które zmienne mają największy wpływ na decyzje modelu

### 2. **SHAP Summary Plot**
- Szczegółowa analiza wpływu każdej cechy
- **Kierunek wpływu** (pozytywny lub negatywny)
- **Rozkład wartości** dla wszystkich klientów

### 3. **SHAP Bar Plot**
- Prosty ranking cech według średniego absolutnego wpływu
- Świetny do prezentacji dla managementu

### 4. **Force Plot**
- Analiza **pojedynczego klienta**
- Wyjaśnienie: "Dlaczego ten konkretny klient ma wysokie ryzyko?"

### 5. **Waterfall Plot** 🌊
- Bardziej czytelna alternatywa dla Force Plot
- Pokazuje krok po kroku, jak każda cecha zmienia predykcję
- Różnica od wartości bazowej do finalnej predykcji

## 🎁 Kluczowe wnioski biznesowe

### 🔴 Grupa WYSOKIEGO ryzyka:

1. **Nowi klienci (tenure < 6 miesięcy)** ⭐ NAJWAŻNIEJSZA CECHA
   - Problem: Krótki tenure drastycznie zwiększa ryzyko odejścia
   - Działanie: Program welcome, częsty kontakt, rabaty w pierwszych miesiącach

2. **Wysokie MonthlyCharges**
   - Problem: Wysoka cena irytuje klientów (czerwone punkty w Summary Plot)
   - Działanie: Więcej value za tę samą cenę, targetowane rabaty dla high-risk

3. **Niskie TotalCharges**
   - Problem: Niskie TotalCharges = krótki staż = brak lojalności
   - Działanie: Budowanie długoterminowej relacji (programy lojalnościowe)

4. **Umowy miesięczne (Month-to-month)**
   - Problem: Brak zobowiązania = łatwe odejście
   - Działanie: Zachęty do rocznych/2-letnich kontraktów

5. **Fiber optic + Electronic check**
   - Problem: Wysokie oczekiwania + mało wygodna metoda płatności
   - Działanie: Specjalna obsługa, edukacja o korzyściach, zacheta do automatycznych płatności

## 💼 Strategia retencyjna

### Krok 1: Identyfikacja
- Model przewiduje ryzyko churn dla każdego klienta

### Krok 2: Segmentacja
- SHAP wyjaśnia **DLACZEGO** klient jest zagrożony

### Krok 3: Akcja
- Dedykowane oferty dla każdej grupy ryzyka

### Krok 4: Monitoring
- Śledzenie efektywności działań
- A/B testing różnych strategii

## 📈 Przykładowe działania

### Dla nowych klientów:
- ✅ Welcome pack z instrukcjami
- ✅ Dedykowany contact person przez pierwsze 3 miesiące
- ✅ Rabat w 2. miesiącu: "Zostań z nami!"

### Dla umów miesięcznych:
- ✅ 15% rabatu za roczny kontrakt
- ✅ 25% rabatu za 2-letni kontrakt
- ✅ Dodatkowe usługi za darmo (HBO, więcej GB)

### Dla klientów z wysokimi opłatami:
- ✅ Targetowane rabaty tylko dla high-risk
- ✅ Upgrade pakietu (więcej za tę samą cenę)
- ✅ Bonus points w programie lojalnościowym

## 🔬 Techniczne szczegóły

### Feature Importance
- **Metoda:** Permutation Importance lub model-specific (np. coef_ dla regresji)
- **Interpretacja:** Im wyższa wartość, tym większy wpływ na predykcję

### SHAP Values
- **Metoda:** Shapley values z teorii gier
- **Zalety:** 
  - Pokazuje kierunek wpływu (+ lub -)
  - Suma SHAP values = różnica między predykcją a base value
  - Teoretycznie uzasadnione (Shapley values)
- **Wady:**
  - Obliczenia mogą być wolne dla dużych zbiorów
  - Wymaga próbkowania dla bardzo dużych danych

### Typy eksplanerów SHAP:
- **LinearExplainer** → szybki dla modeli liniowych
- **TreeExplainer** → szybki dla modeli drzewiastych
- **KernelExplainer** → uniwersalny, działa z każdym modelem, wolniejszy ✅ Użyty w projekcie

### Waterfall Plot vs Force Plot:
- **Force Plot**: poziomy, wszystkie cechy na jednym wykresie (może być zatloczone)
- **Waterfall Plot**: pionowy, krok po kroku, łatwiejszy do zrozumienia ✅ Zalecany

## 📖 Co dalej?

### Rozszerzenia projektu:
1. **LIME** - alternatywna metoda interpretacji
2. **Dependence plots** - analiza interakcji między cechami
3. **Fairness analysis** - sprawdzenie, czy model jest bezstronny
4. **ICE plots** - analiza indywidualnych warunkowych oczekiwań

### Wdrożenie produkcyjne:
1. Integracja z systemem CRM
2. Automatyzacja predykcji (daily batch)
3. Dashboard z wynikami SHAP
4. Monitoring efektywności działań retencyjnych

## 📚 Materiały dodatkowe

### SHAP:
- [SHAP dokumentacja](https://shap.readthedocs.io/)
- [SHAP paper (NIPS 2017)](https://arxiv.org/abs/1705.07874)

### Interpretable ML:
- [Interpretable Machine Learning book](https://christophm.github.io/interpretable-ml-book/)
- [Google's ML Explainability](https://cloud.google.com/explainable-ai)

### PyCaret:
- [PyCaret dokumentacja](https://pycaret.org/)
- [PyCaret Classification Guide](https://pycaret.gitbook.io/docs/get-started/functions/classification)

## 🎯 Wymagania biznesowe vs techniczne

| Wymaganie biznesowe | Rozwiązanie techniczne |
|---------------------|------------------------|
| "Dlaczego ten klient odchodzi?" | SHAP Force Plot |
| "Które cechy są najważniejsze?" | Feature Importance + SHAP Bar Plot |
| "Jak wpływa długość kontraktu?" | SHAP Summary Plot (analiza Contract) |
| "Na kogo się skupić?" | Model scoring + SHAP segmentacja |
| "Jakie działania podjąć?" | Interpretacja SHAP → rekomendacje |

## ⚖️ Zgodność z regulacjami

### GDPR - Prawo do wyjaśnienia
- ✅ SHAP dostarcza **wyjaśnialnych** predykcji
- ✅ Możliwość pokazania klientowi, dlaczego otrzymał daną ofertę
- ✅ Transparentność algorytmów ML

## 🎉 Podsumowanie

Ten projekt pokazuje, że **modele ML nie muszą być czarnymi skrzynkami**. 

Dzięki SHAP i Feature Importance możemy:
- ✅ Zrozumieć decyzje modelu
- ✅ Znaleźć przyczyny problemów biznesowych
- ✅ Podjąć konkretne, data-driven działania
- ✅ Budować zaufanie do AI w organizacji

**Pamiętaj:** Interpretacja modelu jest równie ważna jak jego accuracy! 🚀

---

## 👤 Autor

Projekt stworzony jako część portfolio Machine Learning.

Data: Styczeń 2026

## 📝 Licencja

Ten projekt jest dostępny do celów edukacyjnych.
