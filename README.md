# MLP-Tree Hybrid Classifier

Ten projekt implementuje innowacyjny model hybrydowy do klasyfikacji danych tabelarycznych, łączący nieliniową ekstrakcję cech przez sieć neuronową z efektywnością drzew decyzyjnych.

## 🚀 Opis Projektu
Głównym założeniem systemu jest wykorzystanie sieci **MLP (Multi-Layer Perceptron)** jako enkodera, który uczy się optymalnej reprezentacji danych (embeddingów). Na tak przygotowanych cechach trenowane jest **drzewo decyzyjne**, co pozwala na uzyskanie modelu łączącego zalety głębokiego uczenia z interpretowalnością struktur drzewiastych.

### Główne cechy:
* **Hybrydowa architektura**: Połączenie `MLPEncoder` z `DecisionTreeClassifier`.
* **Porównanie modeli**: Narzędzia do zestawiania wyników hybrydy z klasycznym MLP oraz pojedynczym drzewem decyzyjnym.
* **Wszechstronność**: Obsługa różnorodnych zbiorów danych tabelarycznych (np. Wine, Cancer, Adult, Heart).
* **Analiza statystyczna**: Możliwość uruchamiania wielokrotnych powtórzeń eksperymentów w celu wyznaczenia średniej dokładności i odchylenia standardowego.

## 🏗️ Szczegółowa Architektura Systemu

System opiera się na modularnych komponentach, które mogą działać samodzielnie lub w układzie hybrydowym. Poniżej przedstawiono implementację kluczowych klas.

### 1. Model MLP (Klasyfikator)
Klasa `MLP` to pełna sieć neuronowa służąca do pre-treningu reprezentacji. Składa się z enkodera (`MLPEncoder`) oraz głowicy klasyfikującej (`classifier`). Kluczową cechą jest metoda `forward`, która zwraca zarówno logity (do klasyfikacji), jak i embeddingi (do drzewa).

**Kluczowe metody:**
* `forward(X)`: Przepływ danych przez sieć.
* `predict(X)`: Wykonuje predykcję klas w trybie ewaluacji (bez liczenia gradientów).

```python
# src/models/mlp.py

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, num_layers, num_classes):
        super(MLP, self).__init__()
        # Enkoder uczący się reprezentacji
        self.encoder = MLPEncoder(input_dim, hidden_dim, embedding_dim, num_layers)
        # Warstwa klasyfikująca (używana tylko podczas treningu sieci)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, X):
        """
        Zwraca zarówno wynik klasyfikacji (logits), jak i reprezentację wewnętrzną (embeddings).
        """
        embeddings = self.encoder(X)
        logits = self.classifier(embeddings)
        return logits, embeddings

    def predict(self, X):
        """
        Zwraca predykcje klas dla danych wejściowych X.
        """
        self.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X)
            logits, _ = self(X_t)
            preds = torch.argmax(logits, dim=1)
        return preds.cpu().numpy()
```

### 2. Model Drzewa Decyzyjnego (Wrapper)
Klasa `DecisionTreeModel` stanowi wrapper na `DecisionTreeClassifier` z biblioteki scikit-learn. Jej głównym celem jest ujednolicenie interfejsu (metody `fit`, `predict`, `score`) względem modelu MLP, co pozwala na łatwą wymianę modeli w systemie eksperymentalnym.

**Kluczowe metody:**
* `fit(X, y)`: Trenuje klasyczne drzewo decyzyjne na dostarczonych danych.
* `predict(X)`: Zwraca predykcje klas dla danych wejściowych.
* `score(X, y)`: Oblicza dokładność (accuracy) modelu.

```python
# src/models/decision_tree_model.py

class DecisionTreeModel:
    """
    A wrapper for the Decision Tree Classifier from scikit-learn.
    """

    def __init__(self, max_depth: int = None, random_state = None):
        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=random_state
        )

    def fit(self, X, y):
        """
        Fit the Decision Tree model.
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Predict using the Decision Tree model.
        """
        return self.model.predict(X)

    def score(self, X, y):
        """
        Score the Decision Tree model.
        """
        return self.model.score(X, y) 
   ```
### 3. Model Hybrydowy (Integracja)
Klasa `HybridModel` realizuje główną koncepcję projektu – łączenie reprezentacji neuronowej z decyzyjnością drzewa. Działa jako orkiestrator: najpierw trenuje sieć MLP, aby nauczyła się użytecznych cech, następnie ekstrahuje te cechy (embeddingi) i wykorzystuje je do wytrenowania drzewa decyzyjnego.

**Kluczowe metody:**
* `fit(X, y)`: Zarządza sekwencyjnym procesem uczenia (MLP $\to$ Ekstrakcja $\to$ Drzewo).
* `predict(X)`: Przekształca nowe dane przez zamrożony enkoder i klasyfikuje wynik drzewem.
* `_get_embeddings(X)`: Metoda wewnętrzna transformująca surowe dane na wektory z warstwy ukrytej.

```python
# src/models/hybrid_model.py

class HybridModel:
    """
    Hybrid model: MLP (Encoder) + Decision tree.
    Training and extraction logic is included in this class.
    """

    def fit(self, X, y):
        """
        Główna metoda ucząca:
        1. Inicjalizacja i trening MLP na surowych danych.
        2. Ekstrakcja embeddingów.
        3. Trening drzewa na uzyskanych embeddingach.
        """
        # Krok 1: Trening MLP (Pre-training)
        self._set_mlp()
        trainer = MLPTrainer(self.lr, self.epochs)
        trainer.set_model(self.mlp)
        trainer.fit(X, y)

        # Krok 2: Ekstrakcja cech (embeddingów) z sieci
        X_emb = self._get_embeddings(X)

        # Krok 3: Trening drzewa na embeddingach
        self.tree = DecisionTreeModel(
            max_depth=self.tree_max_depth,
            random_state=self.random_state
        )
        self.tree.fit(X_emb, y)

        self.is_fitted = True
        return self

    def predict(self, X):
        """
        Predykcja klas w podejściu hybrydowym.
        """
        if not self.is_fitted:
            raise Exception("Train model first")

        # Transformacja: Surowe dane -> Embeddingi -> Decyzja Drzewa
        X_emb = self._get_embeddings(X)
        return self.tree.predict(X_emb)
   ```



## 📂 Struktura Plików
| Plik | Opis |
| :--- | :--- |
| `hybrid_model.py` | Implementacja klasy `HybridModel` zarządzającej przepływem danych między MLP a drzewem. |
| `mlp_encoder.py` | Definicja architektury sieci neuronowej (warstwy liniowe + ReLU). |
| `mlp.py` | Model MLP łączący enkoder z warstwą klasyfikującą. |
| `MLPTrainer.py` | Klasa odpowiedzialna za proces uczenia sieci neuronowej (optymalizator Adam, funkcja CrossEntropy). |
| `decision_tree_model.py` | Wrapper dla modelu `DecisionTreeClassifier` z biblioteki scikit-learn. |
| `dataset_factory.py` | Fabryka odpowiedzialna za ładowanie i podział zbiorów danych. |
| `run_experiment.py` | Skrypty do uruchamiania testów porównawczych i zbierania metryk. |
| `experiment_config.py` | Definicje struktur konfiguracyjnych (hiperparametry modeli). |
| `experiments_list.py` | Lista predefiniowanych eksperymentów dla konkretnych zbiorów danych. |

## 📊 Metodologia Badań
Każdy eksperyment dostarcza szczegółowych informacji o:
* **Accuracy (Train/Test)**: Skuteczność modelu na danych znanych i nowych.
* **Generalization Gap**: Miara overfittingu (różnica między zbiorem treningowym a testowym).
* **Training Time**: Czas potrzebny na pełne wytrenowanie hybrydy w porównaniu do modeli bazowych.

## 🛠️ Instrukcja Użycia
1. Zdefiniuj parametry eksperymentu w `experiments_list.py`.
2. Wykorzystaj `run_experiment_avg(cfg)`, aby przeprowadzić wielokrotne testy i otrzymać uśrednione wyniki.
3. (Opcjonalnie) Włącz `do_plots=True`, aby wygenerować wizualizacje porównawcze modeli.
