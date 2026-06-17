# Obserwacje z fazy grupowej MŚ 2026

## Draw Recall Problem
- Model draw recall: 3.57% na datasecie treningowym
- Mundial 2026: 7 remisów z pierwszych 16 meczów (44% vs historyczne 25%)
- Model nie wytypował ani jednego remisu jako faworyta
- Hipoteza do sprawdzenia po fazie grupowej: class_weight adjustment dla klasy "draw"

## Blend Ratio Obserwacje
- Przypadek Australia vs Turcja: model 42.5% Australia, bukmacherzy 20.7%, wygrała Australia 2-0
- Blend 0.15 skrzywdził predykcję — model miał rację, rynek się mylił
- Hipoteza: dla meczów z H2H history rozważyć wyższy blend ratio

## Model vs Bukmacherzy
- Côte d'Ivoire vs Ecuador: brak kursów bukmacherskich, model sam trafił (home_win)
- Dni z wyraźnymi faworytami (>60% finalna): model radzi sobie dobrze
- Dni z wyrównanymi meczami: draw recall problem dominuje

## Name Mapping Issues
- "Cabo Verde" vs "Cape Verde" — niespójność między plikami
- "Iran" vs "IR Iran" — niespójność w results.json
- "Côte d'Ivoire" — web search nie znajduje kursów, alias "Ivory Coast" do dodania

## Accuracy Timeline
- Po 4 meczach: 2/4 = 50%
- Po 8 meczach: 3/8 = 37.5%
- Po 12 meczach: 6/12 = 50%
- Po 16 meczach: 6/16 = 37.5%
- Po 20 meczach: 10/20 = 50%

## Najlepszy dzień
- 17 czerwca: 4/4 = 100% — wszystkie mecze z wyraźnym faworytem >60%

## Do zaimplementowania po fazie grupowej
1. class_weight dla draw w XGBoost retrain
2. Analiza czy zwiększyć blend_ratio dla par z H2H history
3. Alias "Ivory Coast" dla fetch_current_data.py
4. Analiza kalibracji: czy draw probability jest systematycznie zaniżana
