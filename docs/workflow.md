# Fluxo de trabalho

Como o projeto é usado no dia a dia: do setup até o modelo no Model Registry. Os
comandos estão no `Makefile` (`make help` lista todos).

## 1. Preparar a máquina

```bash
make setup             # uv sync + hooks de commit e de push
cp .env.example .env   # opcional; preencha usuário e token do DAGsHub para logar no MLflow remoto
```

Sem o `.env`, o MLflow grava num `mlflow.db` local. Isso serve para testar, mas as runs
não aparecem no DAGsHub. Confira o `.env` antes de treinar algo que vai ser comparado ou
promovido.

## 2. Obter os dados

Dois caminhos, com o mesmo resultado:

- `make etl`: baixa o dataset do Kaggle e gera `data/processed/customer_propensity.parquet`.
- `uv run dvc pull`: baixa do DAGsHub a versão exata registrada no `dvc.lock`. Precisa
  de usuário e token configurados com `dvc remote modify origin --local` (o README mostra
  o comando).

## 3. Mudar algo e treinar

| O que mudou | Onde mudar | Comando |
| --- | --- | --- |
| Hiperparâmetro, seed, split, folds ou threshold | `configs/params.yaml` | `make dvc` |
| Código do ETL, das features ou do treino | `src/` | `make dvc` |
| Espaço de busca do Random Forest | `RF_SEARCH_SPACE` em `src/training/tuning.py` | `make tune-rf` |

O `make dvc` (`dvc repro`) só roda os estágios afetados: mudar o Random Forest não
retreina a regressão logística. No fim, o estágio `promote` aponta o `champion` para a
melhor run, sem criar versão nova se o vencedor não mudou.

A busca (`make tune-rf`) é lenta (25 minutos na última execução) e não altera o modelo
sozinha. Se o resultado for melhor, copie o `best_params` de
`reports/tuning/random_forest.json` para `models.random_forest` em `configs/params.yaml`
(um teste confere que os dois batem) e rode `make dvc`.

## 4. Conferir o resultado

```bash
make dvc-metrics                         # métricas dos dois modelos
uv run dvc metrics diff                  # comparação com o último commit
uv run dvc params diff                   # parâmetros que mudaram
uv run python -m src.training.compare    # ranking das runs no MLflow
```

## 5. Versionar

1. Se os dados ou o ETL mudaram: `uv run dvc push` (precisa de permissão de escrita no
   DAGsHub).
2. Commitar o `dvc.lock`, os JSON de `reports/` e o código juntos. Sem o push do passo
   anterior, os outros recebem um `dvc.lock` que aponta para arquivos que não estão no
   remote.
3. `make lint` e `make test` antes de abrir o PR.

## 6. Branches e PRs

- Branch `<tipo>/<descrição-curta>` a partir da `develop`, com os mesmos tipos dos
  commits (`feat`, `fix`, `data`, `eda`, `exp`, `docs`, `refactor`, `test`, `ci`,
  `build`, `chore`).
- Commits no padrão Conventional Commits, validados pelo hook do commitizen.
- PR para a `develop`. O CI roda lint, testes e o build da imagem Docker, e confere o
  título do PR.
- Release: PR da `develop` para a `main` com merge commit (não squash), para as duas
  branches não divergirem.
- Depois do merge, `cleanup-merged-branch.yml` apaga a branch do PR.

## 7. Rodar no Docker

```bash
make docker-build
make docker-run      # pipeline inteiro, com o .env e data/ montada do host
```

A imagem roda `src/pipeline.py` (ETL, os dois treinos e a promoção), na mesma ordem do
`dvc.yaml`. Ela não tem DVC nem git: dentro dela, a tag `git_sha` das runs fica
`unknown`.
