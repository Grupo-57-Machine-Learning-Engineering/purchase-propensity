.PHONY: help setup install lint format test etl eda pipeline dvc tune-rf dvc-metrics dvc-dag \
	docker-build docker-run pre-commit clean

help: ## Lista os alvos disponíveis
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## Instala dependências (uv) e os hooks do git
	uv sync
	uv run pre-commit install
	cp scripts/reject-push-to-main.sh "$$(git rev-parse --git-path hooks)/pre-push"
	chmod +x "$$(git rev-parse --git-path hooks)/pre-push"

install: ## Instala/atualiza dependências a partir do uv.lock
	uv sync

lint: ## Roda ruff (lint + checagem de formatação)
	uv run ruff check .
	uv run ruff format --check .

format: ## Formata o código com ruff
	uv run ruff format .
	uv run ruff check --fix .

test: ## Roda a suíte de testes (pytest)
	uv run pytest

etl: ## Baixa o dataset do Kaggle e gera data/processed/customer_propensity.parquet
	uv run python -m src.data.pipeline

eda: ## Reexecuta o notebook de EDA (precisa do make etl antes)
	uv run jupyter nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.record_timing=False notebooks/01_eda.ipynb

pipeline: ## Roda o pipeline inteiro sem DVC: ETL, treino e promoção do champion
	uv run python -m src.pipeline

dvc: ## Reproduz o pipeline via DVC (só os estágios desatualizados)
	uv run dvc repro

tune-rf: ## Busca hiperparâmetros do Random Forest (lento; grava reports/tuning/)
	uv run python -m src.training.tuning

dvc-metrics: ## Mostra as métricas dos modelos registradas pelo DVC
	uv run dvc metrics show

dvc-dag: ## Mostra o grafo de dependências do pipeline
	uv run dvc dag

docker-build: ## Constrói a imagem Docker
	docker build -t purchase-propensity .

# MSYS_NO_PATHCONV=1: no Git Bash, impede que o caminho do volume seja convertido (o
# container receberia uma pasta data vazia). No Linux e no Mac, não tem efeito.
docker-run: ## Roda o pipeline no container (usa o .env e grava data/ no host)
	MSYS_NO_PATHCONV=1 docker run --rm --env-file .env -v "$$(pwd)/data:/app/data" purchase-propensity

pre-commit: ## Roda todos os hooks de pre-commit em todos os arquivos
	uv run pre-commit run --all-files

clean: ## Remove caches e artefatos temporários
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
