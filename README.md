# Mundial 2026 — Match Predictor

System predykcji wyników meczów Mistrzostw Świata 2026 łączący model ML wytrenowany na historycznych danych FIFA (1930–2022) z aktualnymi danymi zbieranymi w dzień meczu przez Anthropic API z narzędziem `web_search`. Model XGBoost generuje bazowe prawdopodobieństwa, które następnie są korygowane o kontuzje, morale drużyn i kursy bukmacherskie, a finalna predykcja trafia do raportu HTML.

## Architektura

System składa się z trzech warstw, które działają sekwencyjnie — od offline'owego treningu modelu, przez pobieranie danych w czasie rzeczywistym, aż po finalną predykcję.

**Warstwa 1 — ML pipeline.** Moduł `preprocessing.py` ładuje historyczne dane meczowe z pliku CSV (964 mecze z lat 1930–2022), czyści nazwy drużyn przez `NAME_MAP`, buduje statystyki per-drużyna (win rate ogólny i od 2010+, liczba tytułów, występy w finałach) oraz statystyki head-to-head dla każdej pary. Z tych danych powstaje macierz 16 cech numerycznych — od `rank_diff` i `points_diff` przez `h2h_home_wins` po `home_champion_count`. Moduł `train.py` trenuje XGBoost (z `RandomizedSearchCV`, 30 iteracji, 5-fold CV, optymalizacja `neg_log_loss`) oraz Logistic Regression jako baseline, porównuje oba modele rozszerzonym zestawem metryk i zapisuje najlepsze wersje do `models/`.

**Warstwa 2 — dane dnia meczowego.** Skrypt `fetch_current_data.py` czyta harmonogram turnieju, filtruje mecze zaplanowane na dziś i dla każdego z nich wywołuje Anthropic API (`claude-sonnet-4-6`) z narzędziem `web_search`. Model wyszukuje w internecie i zwraca ustrukturyzowany JSON z kontuzjami, formą ostatnich 5 meczów, morale drużyn, kursami od kilku bukmacherów i najważniejszymi nagłówkami prasowymi. Kursy bukmacherskie są przeliczane na implied probabilities (z usunięciem marży przez normalizację).

**Warstwa 3 — predykcja i output.** Skrypt `predict_today.py` łączy oba źródła sygnału. Najpierw generuje bazowe prawdopodobieństwa z modelu XGBoost, koryguje je o wpływ kontuzji (do ±6 pp) i morale (±3 pp), a następnie blenduje z kursami bukmacherskimi. Blend ratio jest adaptacyjny — 0.15 dla par z historią H2H w danych treningowych, 0.0 dla par bez historii (pełne zaufanie bukamcherom). Finalne prawdopodobieństwa trafiają do `results/results.json` (tracking predykcji) i do wizualnego raportu HTML z paskami prawdopodobieństw oraz tabelą benchmarkową model vs bukmacherzy vs finalna predykcja.

## Stack technologiczny

| Kategoria | Narzędzia |
|---|---|
| Język | Python 3.11+ |
| ML | XGBoost, scikit-learn (Logistic Regression, StandardScaler, RandomizedSearchCV, GridSearchCV) |
| Dane | pandas, NumPy |
| Serializacja | joblib |
| LLM API | Anthropic SDK (`claude-sonnet-4-6` + `web_search`) |
| Env | python-dotenv |
| Środowisko | Ubuntu/WSL 2, VSCode, Claude Code |

## Wyniki modelu (Warstwa 1)

| Model | Accuracy | Log Loss | ROC AUC (macro) |
|---|---|---|---|
| **XGBoost tuned** | 77.2% | **0.364** | **0.944** |
| LR baseline | 82.4% | 0.481 | 0.924 |

XGBoost jest modelem głównym mimo niższej accuracy, ponieważ ma znacząco lepszy log loss (0.364 vs 0.481) i wyższy ROC AUC. W systemie predykcji sportowych kluczowa jest kalibracja prawdopodobieństw — nie tylko *kto* wygra, ale *z jakim prawdopodobieństwem*. Logistic Regression osiąga wyższą accuracy, bo agresywniej stawia na faworyta (ignorując remisy), ale jej szacunki prawdopodobieństw są mniej wiarygodne. XGBoost lepiej modeluje niepewność i edge-case'y, co przekłada się na lepszy blend z kursami bukmacherskimi w Warstwie 3.

## Dane

System korzysta z 5 plików CSV w katalogu `data/raw/` (gitignored — nie są częścią repozytorium):

1. **`matches_1930_2022.csv`** — 964 mecze z historii Mistrzostw Świata (1930–2022): drużyny, wyniki, rundy, daty.
2. **`fifa_ranking_2022-10-06.csv`** — ranking FIFA z okresu MŚ 2022, używany jako proxy historyczne przy treningu.
3. **`fifa_ranking_2026-06-08.csv`** — aktualny ranking FIFA, używany w runtime do cech `rank_diff` i `points_diff`.
4. **`schedule_2026.csv`** — harmonogram meczów MŚ 2026 z datami, drużynami i rundami.
5. **`world_cup.csv`** — dane uzupełniające o turniejach.

Trzy nazwy drużyn wymagają mapowania między źródłami danych (`NAME_MAP` w `preprocessing.py`): United States → USA, Bosnia-Herzegovina → Bosnia and Herzegovina, Cape Verde → Cabo Verde.

## Schemat użycia (każdy dzień meczowy)

```bash
# 1. Pobierz dane dnia meczowego (kontuzje, forma, kursy)
python src/fetch_current_data.py

# 2. Wygeneruj predykcje i raport HTML
python src/predict_today.py

# 3. Po meczu — zapisz rzeczywisty wynik i zaktualizuj accuracy
python src/update_results.py --home "Argentina" --away "Canada" --result home_win
```

Jednorazowo po fazie grupowej (gdy dostępne są wyniki meczów z 2026) można przetrenować model na poszerzonym zbiorze danych:

```bash
python src/update_model.py
```

## Struktura projektu

```
mundial-match-predictor/
├── src/
│   ├── preprocessing.py      # Ładowanie CSV, name mapping, team stats, H2H, feature matrix
│   ├── train.py               # Trening XGBoost + LR, hyperparameter tuning, metryki, zapis modeli
│   ├── fetch_current_data.py  # Anthropic API + web_search → JSON z danymi dnia meczowego
│   ├── predict_today.py       # Blend model + bukmacherzy, korekty, HTML output, tracking
│   ├── update_results.py      # Zapis rzeczywistych wyników, obliczenie accuracy
│   └── update_model.py        # Retrenowanie XGBoost na poszerzonym zbiorze (+ mecze 2026)
├── data/
│   ├── raw/                   # Pliki CSV (gitignored)
│   └── processed/             # Feature matrix, JSON z danymi dnia meczowego
├── models/                    # model.pkl (XGBoost), baseline_model.pkl (LR) — gitignored
├── outputs/                   # daily_predictions.html — raport wizualny
├── results/                   # results.json — tracking predykcji i accuracy
├── notebooks/                 # Eksploracja danych (opcjonalnie)
├── .env.template              # Szablon konfiguracji API key
├── requirements.txt           # Zależności Python
└── setup.sh                   # Automatyczny setup projektu
```

## Setup

```bash
# Sklonuj repozytorium
git clone https://github.com/JarzaboKJ/mundial-match-predictor.git
cd mundial-match-predictor

# Utwórz i aktywuj wirtualne środowisko
python3 -m venv venv
source venv/bin/activate

# Zainstaluj zależności
pip install -r requirements.txt

# Skonfiguruj klucz API
cp .env.template .env
# Wstaw swój ANTHROPIC_API_KEY w pliku .env

# Skopiuj pliki CSV do data/raw/
cp /ścieżka/do/matches_1930_2022.csv data/raw/
cp /ścieżka/do/fifa_ranking_*.csv data/raw/
cp /ścieżka/do/schedule_2026.csv data/raw/
cp /ścieżka/do/world_cup.csv data/raw/

# Wytrenuj model (jednorazowo)
python src/train.py
```

Alternatywnie: `bash setup.sh` wykonuje kroki 2–4 automatycznie.

---

Projekt zbudowany podczas FIFA World Cup 2026 jako praktyczne zastosowanie ML i LLM API w czasie rzeczywistym.
