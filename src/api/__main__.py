"""Sobe a API com uvicorn: `python -m src.api` (ou `make api`).

- `API_HOST`: interface de rede. Padrão `127.0.0.1` (só a própria máquina); a
  imagem Docker usa `0.0.0.0` para aceitar conexões de fora do container.
- `PORT`: porta. Padrão 8000. É o nome que serviços de deploy como o Render usam
  para informar a porta em que o app deve escutar.
"""

import uvicorn

from src.config import env


def main() -> None:
    """Ponto de entrada de linha de comando."""
    uvicorn.run("src.api.app:app", host=env("API_HOST", "127.0.0.1"), port=int(env("PORT", "8000")))


if __name__ == "__main__":
    main()
