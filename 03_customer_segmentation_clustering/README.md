# 🛍️ Customer Segmentation - Segmentacja Klientów

## 📋 Opis projektu

Projekt segmentacji klientów centrum handlowego wykorzystujący algorytm **K-Means** i bibliotekę **PyCaret**. Model automatycznie dzieli klientów na grupy o podobnych cechach, co umożliwia targetowane kampanie marketingowe.

## 🎯 Cel biznesowy

- Identyfikacja grup klientów o podobnych zachowaniach zakupowych
- Personalizacja ofert marketingowych dla każdej grupy
- Optymalizacja strategii sprzedaży i programów lojalnościowych
- Lepsze zrozumienie bazy klientów

## 📊 Dane

**Źródło:** Mall Customers Dataset

**Cechy:**
- `CustomerID` - Unikalny identyfikator klienta
- `Gender` - Płeć
- `Age` - Wiek klienta
- `Annual Income (k$)` - Roczny dochód w tysiącach dolarów
- `Spending Score (1-100)` - Punkty wydatków przyznawane przez centrum handlowe

**Cechy użyte do klastrowania:**
- Wiek
- Roczny dochód
- Punkty wydatków

## 🔧 Technologie

- **Python 3.8+**
- **PyCaret** - AutoML dla klastrowania
- **Scikit-learn** - K-Means, metryki
- **Pandas** - Przetwarzanie danych
- **Matplotlib / Seaborn** - Wizualizacje
- **NumPy** - Operacje numeryczne

## 📁 Struktura projektu

```
03_customer_segmentation_clustering/
│
├── data/
│   ├── Mall_Customers.csv              # Dane źródłowe
│   └── customers_with_clusters.csv     # Dane z przypisanymi klastrami
│
├── models/
│   └── customer_segmentation_model.pkl # Zapisany model K-Means
│
├── customer_segmentation.ipynb         # Notebook z analizą (szczegółowy)
├── customer_segmentation.py            # Skrypt Python (wersja produkcyjna)
└── README.md                           # Ten plik
```

## 📈 Proces analizy

1. **Wczytanie danych** - Załadowanie danych klientów z CSV
2. **Eksploracja danych** - Analiza rozkładów i statystyk
3. **Przygotowanie danych** - Wybór cech numerycznych, normalizacja
4. **Inicjalizacja PyCaret** - Automatyczne przygotowanie pipeline'u
5. **Dobór liczby klastrów** - Metoda Elbow
6. **Trenowanie modelu K-Means** - Utworzenie 5 klastrów
7. **Przypisanie klastrów** - Etykietowanie klientów
8. **Analiza klastrów** - Interpretacja biznesowa grup
9. **Wizualizacje** - Wykresy 2D, 3D, box plots
10. **Zapis modelu** - Zapisanie do późniejszego użycia

## 🎯 Wyniki segmentacji

Model dzieli klientów na **5 głównych grup**:

### KLASTER Cluster 0 (40 klientów, 20.0%):
   • Średni wiek: 33 lat
   • Średni dochód roczny: $86k
   • Punkty wydatków: 82/100

   📝 KIM SĄ CI KLIENCI?
   → VIP / PREMIUM - Wysokie dochody i wysokie wydatki
   💡 Strategia: Produkty luksusowe, obsługa VIP, ekskluzywne wydarzenia

--------------------------------------------------------------------------------

### KLASTER Cluster 1 (47 klientów, 23.5%):
   • Średni wiek: 56 lat
   • Średni dochód roczny: $54k
   • Punkty wydatków: 49/100

   📝 KIM SĄ CI KLIENCI?
   → PRZECIĘTNI KLIENCI - Średnie dochody, średnie wydatki
   💡 Strategia: Standardowe oferty, programy lojalnościowe

--------------------------------------------------------------------------------

### KLASTER Cluster 2 (54 klientów, 27.0%):
   • Średni wiek: 25 lat
   • Średni dochód roczny: $41k
   • Punkty wydatków: 62/100

   📝 KIM SĄ CI KLIENCI?
   → VIP / PREMIUM - Wysokie dochody i wysokie wydatki
   💡 Strategia: Produkty luksusowe, obsługa VIP, ekskluzywne wydarzenia

--------------------------------------------------------------------------------

### KLASTER Cluster 3 (39 klientów, 19.5%):
   • Średni wiek: 40 lat
   • Średni dochód roczny: $86k
   • Punkty wydatków: 19/100

   📝 KIM SĄ CI KLIENCI?
   → BOGACI OSZCZĘDNI - Wysokie dochody, ale ostrożne wydatki
   💡 Strategia: Produkty premium z uzasadnioną wartością, ekskluzywne oferty

--------------------------------------------------------------------------------

### KLASTER Cluster 4 (20 klientów, 10.0%):
   • Średni wiek: 46 lat
   • Średni dochód roczny: $27k
   • Punkty wydatków: 18/100

   📝 KIM SĄ CI KLIENCI?
   → GRUPA OSZCZĘDNA - Niskie dochody, małe wydatki
   💡 Strategia: Oferty promocyjne, rabaty, karty lojalnościowe

--------------------------------------------------------------------------------

## 📊 Metryki modelu

- **Liczba klastrów:** 5 (wybrane metodą Elbow)
- **Algorytm:** K-Means
- **Normalizacja:** zscore (automatyczna w PyCaret)
- **Silhouette Score:** ~0.42 (dobra separacja klastrów)

## 🔮 Predykcja dla nowych klientów

Model może przypisać nowego klienta do odpowiedniej grupy:

```python
# Przykład użycia
new_customer = {
    'Age': 28,
    'Annual Income (k$)': 75,
    'Spending Score (1-100)': 80
}

predicted_cluster = predict_new_customer(model, **new_customer)
# Zwraca: Cluster 0 (VIP)
```

## 💡 Wnioski biznesowe

1. **Personalizacja** - Różne grupy wymagają różnych strategii marketingowych
2. **Optymalizacja budżetu** - Skupienie zasobów na najbardziej dochodowych segmentach
3. **Retencja** - Programy lojalnościowe dostosowane do potrzeb grup
4. **Cross-selling** - Oferty produktowe dopasowane do profilu klienta
5. **Komunikacja** - Targetowane kampanie e-mail/SMS do konkretnych segmentów

### Data wpisu
03.01.2026
