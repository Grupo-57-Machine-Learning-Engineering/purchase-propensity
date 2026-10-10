"""Configuração central: paths do projeto.

Fonte única de verdade para valores compartilhados entre as etapas do
pipeline. Sem lógica de negócio, apenas constantes.
"""

from pathlib import Path

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = PROJECT_ROOT / "data"
MODELS_DIR: Path = PROJECT_ROOT / "models"
