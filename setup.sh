#!/usr/bin/env bash
# =============================================================================
# setup.sh — mundial-match-predictor
# Uruchom: bash setup.sh
# Zakłada, że jesteś w ~/projekty/mundial-match-predictor
# =============================================================================

set -euo pipefail   # wyjdź przy błędzie, niezdefiniowanej zmiennej, błędzie w pipe

# ── Kolory do logowania ───────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${GREEN}[✓]${NC} $*"; }
info() { echo -e "${CYAN}[→]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
fail() { echo -e "${RED}[✗]${NC} $*"; exit 1; }

echo ""
echo -e "${CYAN}=================================================${NC}"
echo -e "${CYAN}  mundial-match-predictor — project setup${NC}"
echo -e "${CYAN}=================================================${NC}"
echo ""

# ── Sprawdź, że jesteśmy w odpowiednim folderze ───────────────────────────────
EXPECTED_DIR="mundial-match-predictor"
CURRENT_DIR=$(basename "$PWD")

if [[ "$CURRENT_DIR" != "$EXPECTED_DIR" ]]; then
    fail "Uruchom skrypt z folderu '${EXPECTED_DIR}'. Aktualny folder: $PWD"
fi
log "Folder projektu: $PWD"

# =============================================================================
# KROK 1 — Struktura folderów
# =============================================================================
info "Tworzę strukturę folderów..."

DIRS=(
    "data/raw"
    "data/processed"
    "models"
    "src"
    "outputs"
    "results"
    "notebooks"
)

for DIR in "${DIRS[@]}"; do
    mkdir -p "$DIR"
    # Dodaj .gitkeep, żeby puste foldery trafiły do gita
    touch "${DIR}/.gitkeep"
done

log "Struktura folderów gotowa."

# =============================================================================
# KROK 2 — requirements.txt
# =============================================================================
info "Tworzę requirements.txt..."

cat > requirements.txt << 'EOF'
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
joblib>=1.3.0
anthropic>=0.28.0
requests>=2.31.0
python-dotenv>=1.0.0
matplotlib>=3.7.0
seaborn>=0.13.0
tqdm>=4.66.0
EOF

log "requirements.txt gotowy."

# =============================================================================
# KROK 3 — .gitignore
# =============================================================================
info "Tworzę .gitignore..."

cat > .gitignore << 'EOF'
# Sekrety i środowisko
.env
venv/
.venv/

# Dane surowe (duże pliki, nie do repozytorium)
data/raw/

# Modele (binarki generowane przez pipeline)
models/
*.pkl

# Python cache
__pycache__/
*.pyc
*.pyo
*.pyd
.Python

# Outputy i wyniki (generowane runtime)
outputs/
results/

# Jupyter
.ipynb_checkpoints/
notebooks/.ipynb_checkpoints/

# Systemy plików
.DS_Store
Thumbs.db

# Build / dist
dist/
build/
*.egg-info/
*.egg

# IDE
.vscode/settings.json
.idea/
EOF

log ".gitignore gotowy."

# =============================================================================
# KROK 4 — .env.template
# =============================================================================
info "Tworzę .env.template..."

cat > .env.template << 'EOF'
# =============================================================================
# mundial-match-predictor — konfiguracja środowiska
# =============================================================================
# Skopiuj ten plik jako .env i wstaw swój klucz:
#   cp .env.template .env
# Klucz API uzyskasz na: https://console.anthropic.com
# NIGDY nie commituj pliku .env do repozytorium!
# =============================================================================

ANTHROPIC_API_KEY=your_api_key_here
EOF

log ".env.template gotowy."

# =============================================================================
# KROK 5 — Wirtualne środowisko + instalacja zależności
# =============================================================================
info "Sprawdzam Python 3..."
python3 --version || fail "Python 3 nie jest zainstalowany lub niedostępny w PATH."

info "Tworzę venv..."
python3 -m venv venv
log "venv utworzony."

info "Aktywuję venv i instoluję zależności (może chwilę potrwać)..."

# Instalacja wewnątrz venv — wywołujemy pip przez ścieżkę bezpośrednią
# (nie możemy `source` w subshell i mieć efektów w rodzicu, więc używamy venv/bin/pip)
VENV_PIP="venv/bin/pip"
VENV_PYTHON="venv/bin/python"

"$VENV_PIP" install --upgrade pip --quiet
"$VENV_PIP" install -r requirements.txt --quiet

log "Zależności zainstalowane."

# =============================================================================
# KROK 6 — git init + initial commit
# =============================================================================
info "Inicjalizuję repozytorium git..."

if [[ -d ".git" ]]; then
    warn ".git już istnieje — pomijam git init."
else
    git init
fi

# Skonfiguruj git lokalnie, jeśli user.email niezdefiniowany (częste w WSL)
if ! git config user.email > /dev/null 2>&1; then
    warn "Brak git user.email — ustawiam wartości placeholder (zmień je później)."
    git config user.email "dev@mundial-predictor.local"
    git config user.name "Mundial Predictor"
fi

git add .
git commit -m "chore: initial project setup"

log "Repozytorium zainicjalizowane, initial commit gotowy."

# =============================================================================
# KROK 7 — Weryfikacja
# =============================================================================
echo ""
echo -e "${CYAN}=================================================${NC}"
echo -e "${CYAN}  Weryfikacja${NC}"
echo -e "${CYAN}=================================================${NC}"
echo ""

# Struktura folderów
info "Struktura projektu (2 poziomy):"
if command -v tree &> /dev/null; then
    tree -L 2 --dirsfirst -a -I ".git"
else
    warn "'tree' niedostępne — używam find:"
    find . -not -path "./.git/*" -not -name ".git" | \
        sort | \
        awk -F/ '{
            depth = NF - 2;
            indent = "";
            for (i=0; i<depth; i++) indent = indent "  ";
            print indent "├── " $NF
        }'
fi

echo ""

# Git log
info "Historia commitów:"
git log --oneline

echo ""

# Git status — .env i data/raw/ powinny być ignorowane
info "Git status (sprawdzam ignorowane ścieżki):"
git status

echo ""

# Sprawdź jawnie, czy .env i data/raw są ignorowane
touch .env 2>/dev/null || true   # stwórz tymczasowo żeby przetestować
IGNORED_ENV=$(git check-ignore -v .env 2>/dev/null && echo "TAK" || echo "NIE")
IGNORED_RAW=$(git check-ignore -v data/raw 2>/dev/null && echo "TAK" || echo "NIE")
rm -f .env 2>/dev/null || true   # usuń plik testowy

echo -e "  .env ignorowany przez git:      ${GREEN}${IGNORED_ENV}${NC}"
echo -e "  data/raw/ ignorowany przez git: ${GREEN}${IGNORED_RAW}${NC}"

echo ""

# Wersje kluczowych pakietów
info "Zainstalowane pakiety (kluczowe):"
"$VENV_PIP" list 2>/dev/null | grep -E "pandas|xgboost|anthropic|scikit|numpy|joblib"

echo ""
echo -e "${GREEN}=================================================${NC}"
echo -e "${GREEN}  Setup zakończony pomyślnie!${NC}"
echo -e "${GREEN}=================================================${NC}"
echo ""
echo -e "  Następne kroki:"
echo -e "  1. ${CYAN}source venv/bin/activate${NC}   — aktywuj venv w bieżącym terminalu"
echo -e "  2. ${CYAN}cp .env.template .env${NC}      — stwórz plik .env"
echo -e "  3. Wstaw klucz API do ${CYAN}.env${NC}     — z console.anthropic.com"
echo -e "  4. Skopiuj pliki CSV do    ${CYAN}data/raw/${NC}"
echo ""