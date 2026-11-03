# Imagem do pipeline do purchase-propensity: ETL, treino e promoção no Model Registry.
#
# Build:  docker build -t purchase-propensity .
# Run:    docker run --rm --env-file .env -v "$(pwd)/data:/app/data" purchase-propensity
FROM python:3.11-slim-bookworm

# uv fixado na mesma versão usada no CI (.github/workflows/ci.yml).
COPY --from=ghcr.io/astral-sh/uv:0.11.13 /uv /usr/local/bin/uv

# - UV_PYTHON_DOWNLOADS=never: usa o Python 3.11 da imagem, sem baixar outro.
# - UV_LINK_MODE=copy: o cache do uv fica num mount de build, sem hardlink.
# - PATH: o venv do uv vem primeiro, então `python` já é o do projeto.
# - KAGGLEHUB_CACHE e USER: deixam `docker run --user <uid>` funcionar (uid sem
#   home e sem entrada no passwd), útil em bind mount no Linux.
# - GIT_PYTHON_REFRESH=quiet: a imagem não tem git, então o MLflow não consegue
#   registrar o git_sha; isto só silencia o aviso longo que ele imprime por isso.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    KAGGLEHUB_CACHE=/tmp/kagglehub \
    USER=app \
    GIT_PYTHON_REFRESH=quiet \
    PATH="/app/.venv/bin:$PATH"

# Usuário sem privilégios: o pipeline não precisa de root.
RUN useradd --create-home --uid 1000 --shell /usr/sbin/nologin app

WORKDIR /app

# Dependências de produção, sem o grupo dev. Esta camada só é refeita quando
# pyproject.toml ou uv.lock mudam. --locked falha se o lock estiver desatualizado.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --locked --no-dev

COPY configs ./configs
COPY src ./src

# O app escreve em /app (mlflow.db, mlruns/), data/ e models/. O chown em /app
# não é recursivo, para não duplicar o .venv numa camada nova.
RUN mkdir -p data/raw data/processed models \
    && chown app:app /app \
    && chown -R app:app data models

USER app

# Padrão: pipeline de ponta a ponta (src/pipeline.py), na ordem do dvc.yaml.
# Uma etapa só: docker run ... purchase-propensity python -m src.data.pipeline
CMD ["python", "-m", "src.pipeline"]
