# Mundial 2026 — Match Predictor

System predykcji wyników meczów Mistrzostw Świata 2026 łączący model ML wytrenowany na historycznych danych FIFA (1930–2022) z aktualnymi danymi zbieranymi w dzień meczu przez Anthropic API z narzędziem `web_search`. Model XGBoost generuje bazowe prawdopodobieństwa, które są korygowane o kontuzje, formę i morale drużyn, a następnie blendowane z kursami bukmacherskimi. Finalna predykcja trafia do raportu HTML.

## Architektura

System składa się z trzech warstw, które działają sekwencyjnie — od offline'owego treningu modelu, przez pobieranie danych w czasie rzeczywistym, aż po finalną predykcję.

**Warstwa 1 — ML pipeline.** Moduł `preprocessing.py` ładuje historyczne dane meczowe z pliku CSV (964 mecze z lat 1930–2022) i kanonizuje nazwy drużyn przez `NAME_MAP` — łącznie z państwami-sukcesorami (Czechosłowacja→Czechia, RFN→Germany, Zair→Congo DR, ZSRR→Russia, Jugosławia→Serbia), dzięki czemu dorobek historyczny poprawnie łączy się z aktualnym rankingiem FIFA i harmonogramem 2026. Macierz 16 cech (od `rank_diff` po `home_champion_count`) budowana jest **chronologicznie bez leakage**: cechy każdego meczu liczone są wyłącznie z meczów rozegranych wcześniej (poprzednia wersja liczyła statystyki na całym zbiorze, przez co dla 47% par — grających ze sobą tylko raz — cechy H2H kodowały wynik samego meczu). Tytuły mistrzowskie pochodzą z `world_cup.csv` (finały rozstrzygnięte karnymi i turniej 1950 są poprawnie przypisane). Moduł `train.py` trenuje XGBoost (z `RandomizedSearchCV`, 30 iteracji, 5-fold CV, optymalizacja `neg_log_loss`) oraz Logistic Regression jako baseline, ocenia je na **holdoucie czasowym** (trening: 1930–2014, test: turnieje 2018 i 2022), po czym refituje na całości i zapisuje do `models/`.

**Warstwa 2 — dane dnia meczowego.** Skrypt `fetch_current_data.py` czyta harmonogram turnieju, filtruje mecze zaplanowane na dziś i dla każdego z nich wywołuje Anthropic API (`claude-haiku-4-5`) z narzędziem `web_search`. Model wyszukuje w internecie i zwraca ustrukturyzowany JSON z kontuzjami, formą ostatnich 5 meczów, morale drużyn, kursami od kilku bukmacherów i najważniejszymi nagłówkami prasowymi. Odpowiedź LLM traktowana jest jako niezaufane wejście: pola są walidowane i normalizowane (clamp form_score do [-2,2], morale do high/medium/low, kursy muszą być liczbami > 1.0), a nazwy drużyn zawsze pochodzą z harmonogramu, nie z odpowiedzi modelu. Kursy są przeliczane na implied probabilities (z usunięciem marży przez normalizację); brak użytecznych kursów jest jawnie oznaczany flagą `has_bookie_odds`.

**Warstwa 3 — predykcja i output.** Skrypt `predict_today.py` łączy oba źródła sygnału. Najpierw koryguje prawdopodobieństwa modelu XGBoost o wpływ kontuzji (multiplikatywnie, do −15% szansy na wygraną strony), formę (±2 pp za punkt różnicy) i morale (±3 pp) — korekty aplikowane są **na komponencie modelowym przed blendem**, bo bukmacherzy mają te informacje już wycenione w kursach (korygowanie ich kursów liczyłoby ten sam sygnał podwójnie). Następnie skorygowany model jest blendowany z kursami (wagi 0.15/0.85); przy braku kursów finalna predykcja to sam skorygowany model. Wyniki trafiają do `results/results.json` (tracking predykcji) i do raportu HTML z paskami prawdopodobieństw oraz tabelą benchmarkową model vs model+korekty vs bukmacherzy vs finalna predykcja.

## Stack technologiczny

| Kategoria | Narzędzia |
|---|---|
| Język | Python 3.11+ |
| ML | XGBoost, scikit-learn (Logistic Regression, StandardScaler, RandomizedSearchCV, GridSearchCV) |
| Dane | pandas, NumPy |
| Serializacja | joblib |
| LLM API | Anthropic SDK (`claude-haiku-4-5` + `web_search`) |
| Env | python-dotenv |
| Testy | pytest |
| Środowisko | Ubuntu/WSL 2, VSCode, Claude Code |

## Wyniki modelu (Warstwa 1)

Ewaluacja na holdoucie czasowym: trening 1930–2014, test = turnieje 2018 i 2022 (128 meczów).

| Model | Accuracy | Log Loss | ROC AUC (macro) |
|---|---|---|---|
| **XGBoost tuned** | **53.1%** | **1.035** | **0.626** |
| LR tuned | 47.7% | 1.080 | 0.588 |

Te liczby są realistyczne dla predykcji 1X2 w piłce nożnej (losowy strzał = log loss 1.099; topowi bukmacherzy ≈ 0.95–1.00). Wcześniej raportowane metryki (log loss 0.364, ROC AUC 0.944) pochodziły z pipeline'u z target leakage — statystyki drużyn i H2H były liczone na pełnym zbiorze, więc cechy każdego meczu zawierały jego własny wynik. Po naprawie model jest skromnym, ale uczciwym sygnałem — dlatego w Warstwie 3 dostaje wagę 0.15, a kursy bukmacherskie 0.85. XGBoost pozostaje modelem głównym: wygrywa z LR na wszystkich metrykach (kalibracja prawdopodobieństw — log loss i Brier — jest ważniejsza niż sama accuracy przy blendowaniu z kursami).

## Dane

System korzysta z 5 plików CSV w katalogu `data/raw/` (gitignored — nie są częścią repozytorium):

1. **`matches_1930_2022.csv`** — 964 mecze z historii Mistrzostw Świata (1930–2022): drużyny, wyniki, rundy, daty. Plik jest **niemutowalny** — wyniki z 2026 trafiają do osobnego `data/processed/matches_2026.csv`.
2. **`fifa_ranking_2022-10-06.csv`** — ranking FIFA z okresu MŚ 2022, używany jako proxy historyczne przy treningu.
3. **`fifa_ranking_2026-06-08.csv`** — aktualny ranking FIFA, używany w runtime do cech `rank_diff` i `points_diff`.
4. **`schedule_2026.csv`** — harmonogram meczów MŚ 2026 z datami, drużynami i rundami.
5. **`world_cup.csv`** — podsumowania turniejów; źródło prawdy dla `champion_count` i `final_appearances` (mistrz z karnych — 1994/2006/2022 — i format z 1950 są poprawnie obsłużone).

Nazwy drużyn są kanonizowane do nazewnictwa FIFA przez `NAME_MAP` w `preprocessing.py` — poza wariantami pisowni (United States → USA, Bosnia-Herzegovina → Bosnia and Herzegovina, Cape Verde → Cabo Verde) mapowane są państwa-sukcesory: Czech Republic/Czechoslovakia → Czechia, West Germany → Germany, Zaire → Congo DR, Soviet Union → Russia, Yugoslavia/FR Yugoslavia/Serbia and Montenegro → Serbia, Dutch East Indies → Indonesia. Drużyny bez rankingu dostają fallback "gorszy niż najsłabsza notowana drużyna" (a nie rank=100/points=0 jak wcześniej).

## Schemat użycia (każdy dzień meczowy)

```bash
# 1. Pobierz dane dnia meczowego (kontuzje, forma, kursy)
python src/fetch_current_data.py

# 2. Wygeneruj predykcje i raport HTML (opcjonalnie --date YYYY-MM-DD)
python src/predict_today.py

# 3. Po meczu — zapisz rzeczywisty wynik i zaktualizuj accuracy
#    (--date wymagane, gdy te same drużyny grały ze sobą więcej niż raz)
python src/update_results.py --home "Argentina" --away "Canada" --result home_win
```

Testy jednostkowe (logika predykcji, mapowanie nazw, brak leakage):

```bash
python -m pytest tests/
```

Jednorazowo po fazie grupowej (gdy dostępne są wyniki meczów z 2026) można przetrenować model na poszerzonym zbiorze danych:

```bash
python src/update_model.py
```

## Struktura projektu

```
mundial-match-predictor/
├── src/
│   ├── preprocessing.py      # Kanonizacja nazw, chronologiczne cechy (bez leakage), H2H
│   ├── train.py               # Trening XGBoost + LR, tuning, holdout czasowy, zapis modeli
│   ├── fetch_current_data.py  # Anthropic API + web_search → walidowany JSON dnia meczowego
│   ├── predict_today.py       # Korekty na modelu → blend z kursami, HTML, tracking
│   ├── update_results.py      # Zapis rzeczywistych wyników (--date dla rewanżów), accuracy
│   └── update_model.py        # Retrenowanie na historii + data/processed/matches_2026.csv
├── tests/                     # pytest: logika predykcji, name mapping, leakage guard
├── data/
│   ├── raw/                   # Pliki CSV (gitignored, niemutowalne)
│   └── processed/             # Feature matrix, JSON-y dnia meczowego, matches_2026.csv
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
