"""Pacote do purchase-propensity."""

import sys

__version__ = "0.1.0"

# O MLflow imprime emoji ao fechar uma run ("🏃 View run ..."). No Windows, com a
# saída redirecionada (dvc repro, pipe, arquivo), o Python usa cp1252, o print
# levanta UnicodeEncodeError e a run fica presa como RUNNING no servidor.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")
