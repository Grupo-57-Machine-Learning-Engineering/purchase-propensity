# Purchase Propensity

Projeto da Fase 2 da Pós Tech de Machine Learning Engineering - FIAP.

Sistema preditivo para estimar a propensão de compra de um usuário de e-commerce a partir
do comportamento de navegação. O foco é a engenharia de ML: modelo clássico em
Scikit-Learn, dados e pipeline versionados com DVC, experimentos e Model Registry
no MLflow e execução em Docker.

## Dataset

[Customer Propensity to Purchase](https://www.kaggle.com/datasets/benpowis/customer-propensity-to-purchase-data)
(Kaggle, CC0): um dia de visitas a uma loja online fictícia. São 455.401 usuários, 23 flags
de comportamento (cliques, páginas vistas, dispositivo, localização) e o alvo `ordered`
(fez pedido; 4,2% positivos).

`make etl` baixa o dataset com o `kagglehub`, sem precisar de token, e grava
`data/processed/customer_propensity.parquet` com `UserID` renomeado para `user_id` e as
flags 0/1 como booleanos. O ETL não remove colunas: `checked_delivery_detail` e
`saw_checkout` têm correlação alta com `ordered` (0,80 e 0,71), e a EDA decide se elas
entram no modelo. O `testing_sample.csv` do Kaggle não é usado porque vem com `ordered`
zerado.

### EDA

Achados e decisões para a modelagem em [`docs/eda-findings.md`](docs/eda-findings.md); a
análise completa está em [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb). Em resumo:
`sign_in`, `saw_checkout` e `checked_delivery_detail` são vazamento e ficam fora do modelo,
e o carrinho é o melhor sinal de intenção de compra.

## Estrutura

```
purchase-propensity/
├── src/
│   ├── data/         # ETL: extração, transformação e carga
│   ├── features/     # pré-processamento
│   ├── models/       # definição dos modelos
│   ├── training/     # treino, avaliação e registro no MLflow
│   ├── config.py     # paths, dataset e leitura do .env
│   └── logger.py     # logger centralizado (formato e nível únicos)
├── tests/            # pytest
├── data/             # dados locais (fora do git)
├── models/           # artefatos treinados (não versionados no git)
├── configs/          # configurações e hiperparâmetros
├── notebooks/        # EDA e exploração (`NN_tema.ipynb`, ex.: `01_eda.ipynb`)
├── docs/             # documentação do projeto
├── reports/          # métricas do pipeline DVC e resultado da busca de hiperparâmetros
├── .github/          # CI e template de PR
├── .env.example      # modelo do .env (configurações)
├── Dockerfile        # imagem do pipeline (uv, usuário sem privilégios)
├── .dockerignore     # contexto de build sem .env, dados nem venv
├── dvc.yaml          # pipeline DVC (ETL e treino)
├── pyproject.toml    # dependências (uv)
└── Makefile
```

## Setup

Pré-requisitos: [uv](https://docs.astral.sh/uv/) e `make` num shell POSIX (no Windows,
use o Git Bash; o `make` no PowerShell/cmd não roda os alvos). O Python 3.11 é
provisionado pelo uv a partir do `.python-version`.

```bash
make setup        # uv sync + hooks do git
```

As configurações (dataset, pasta de dados, nível de log) ficam no `.env`. Ele é opcional:
sem ele, valem os padrões do `.env.example`. Para mudar algum valor:

```bash
cp .env.example .env
```

### Dados e DVC (DAGsHub)

O dataset bruto (`data/raw/training_sample.csv`) é versionado com DVC e fica no remote do
DAGsHub. Há dois jeitos de ter os dados na máquina:

- `make etl`: baixa do Kaggle (sem token) e gera o parquet. Basta para trabalhar.
- `dvc pull`: baixa a versão exata registrada no DVC. Precisa de credencial do DAGsHub,
  configurada só na sua máquina (`--local` não vai para o git):

  ```bash
  uv run dvc remote modify origin --local user <SEU_USUARIO_DAGSHUB>
  uv run dvc remote modify origin --local password <SEU_TOKEN_DAGSHUB>
  uv run dvc pull
  ```

  O token sai de [dagshub.com/user/settings/tokens](https://dagshub.com/user/settings/tokens).

O pipeline está em `dvc.yaml`: `etl` gera o parquet, e `train_baseline` e
`train_random_forest` treinam os dois modelos, logam no MLflow e gravam as métricas em
`reports/metrics/`. O DVC só reexecuta o estágio cujas dependências mudaram:

```bash
make dvc            # dvc repro: roda o que estiver desatualizado
make dvc-metrics    # dvc metrics show: métricas dos dois modelos lado a lado
make dvc-dag        # grafo dos estágios
```

As métricas em `reports/metrics/` vão para o git, então `uv run dvc metrics diff` compara
o commit atual com o anterior. Quem altera o dataset roda `uv run dvc push` (precisa de
permissão de escrita no repositório do DAGsHub).

## Comandos (Makefile)

| Comando            | O que faz                                  |
| ------------------ | ------------------------------------------ |
| `make setup`       | Instala dependências e hooks do git        |
| `make install`     | Instala dependências a partir do lock      |
| `make lint`        | `ruff check` + checagem de formatação      |
| `make format`      | Formata e auto-corrige com ruff            |
| `make test`        | Roda os testes (pytest)                    |
| `make etl`         | Baixa o dataset e gera `data/processed/`   |
| `make eda`         | Reexecuta o notebook de EDA                |
| `make dvc`         | Reproduz o pipeline via DVC (ETL e treino) |
| `make dvc-metrics` | Mostra as métricas dos modelos (DVC)       |
| `make dvc-dag`     | Exibe o grafo (DAG) de estágios do DVC     |
| `make tune-rf`     | Busca hiperparâmetros do Random Forest     |
| `make pre-commit`  | Roda todos os hooks em todos os arquivos   |
| `make help`        | Lista todos os alvos                       |

## Treino e MLflow

Os dois modelos leem `data/processed/customer_propensity.parquet` (gerado por
`make etl`) e gravam a run no MLflow:

```bash
uv run python -m src.training.baseline
uv run python -m src.training.baseline --run-name baseline-logreg-seed7
uv run python -m src.training.random_forest
```

- **Split:** estratificado pelo alvo `ordered`, com seed fixo (`src/training/split.py`).
- **Vazamento:** `sign_in`, `saw_checkout` e `checked_delivery_detail` ficam fora das
  features por padrão (`LEAKAGE_COLUMNS` em `src/config.py`), conforme a EDA em
  [`docs/eda-findings.md`](docs/eda-findings.md).
- **Métricas:** PR-AUC como principal, mais ROC-AUC, precisão e recall no threshold
  (`src/training/metrics.py`), registradas para treino (`train_*`) e teste (`test_*`).
- **Cross-validation:** k-fold estratificado (`CV_FOLDS` em `src/config.py`) no conjunto
  de treino, com PR-AUC e ROC-AUC médios e desvio padrão entre os folds (`cv_*_mean`,
  `cv_*_std`), para uma estimativa mais estável que o holdout único
  (`src/training/cross_validation.py`).
- **Tempo:** duração do treino e da cross-validation (`fit_duration_seconds`,
  `cv_duration_seconds`, `total_duration_seconds`), para comparar modelos na mesma
  tabela.
- **Reuso:** split, métricas, `fit_and_evaluate` e `cross_validate_model` não dependem do
  modelo, então Random Forest entra trocando só o estimador (`src/training/random_forest.py`).
- **Tracking:** experiment `purchase-propensity`, com as tags `autor`, `git_sha` e
  `dataset_version` (hash do parquet).

### Busca de hiperparâmetros

```bash
make tune-rf    # random search do Random Forest (lento: cerca de 20 minutos)
```

As 20 flags binárias formam só ~4.200 combinações distintas nas 318 mil linhas de treino,
e o Random Forest padrão (árvores sem limite) decorava as raras: PR-AUC de 0,50 no treino
contra 0,37 no teste. A busca (`src/training/tuning.py`) é um random search de 30
candidatos por PR-AUC, nos mesmos 5 folds da cross-validation das runs, focado em
regularização (`min_samples_leaf`, `max_depth`, `max_features`) e no número de árvores.

O resultado vai para `reports/tuning/random_forest.json` (melhores parâmetros e os 10
melhores candidatos) e para uma run do MLflow com a tag `kind=search`, que fica fora do
ranking por não ter modelo. Os melhores parâmetros são copiados para `TUNED_PARAMS` em
`src/training/random_forest.py`: assim o `make dvc` treina sempre o mesmo modelo sem
rodar a busca, que é cara.

### Comparação e Model Registry

```bash
uv run python -m src.training.compare    # tabela comparativa dos runs
uv run python -m src.training.registry   # promove o melhor run a champion
```

- **Comparação:** `src/training/compare.py` ordena por `cv_pr_auc_mean` (mais estável
  que o holdout único) só as runs elegíveis do experimento, e imprime uma tabela com
  modelo, métricas de CV/teste e tempo total. Elegível é a run `FINISHED` treinada sem as
  colunas de vazamento (`params.excluded_columns` igual a `LEAKAGE_COLUMNS`). Uma run que
  cai entre logar a métrica e logar o modelo fica `FAILED` e não teria modelo para
  promover; uma run com vazamento (de referência, ou anterior à exclusão) teria PR-AUC
  artificialmente alta e viraria `champion`.
- **Registry:** `src/training/registry.py` registra o run do topo dessa comparação no
  MLflow Model Registry e aponta o alias `champion` pra ele (aliases, não mais
  "stages", que o MLflow 2.9+/3.x descontinuou).
- **Dagshub:** o registro usa `models:/<model_id>` (o `LoggedModel` do MLflow 3.x criado
  por `mlflow.sklearn.log_model(model, name=..., ...)`), não `runs:/<run_id>/model`.
  Contra o servidor do Dagshub, listar e baixar artefato pelo path do run falha (a
  listagem de artefatos do run não devolve os arquivos); pelo `model_id` funciona.
  Confirmado com teste manual contra o servidor real.
- **skops:** modelos baseados em árvore (Random Forest) precisam de
  `skops_trusted_types=["sklearn.tree._tree.Tree"]` no `log_model` (`src/training/tracking.py`):
  o skops recusa esse tipo por padrão, mas o modelo vem do nosso próprio treino, nunca
  de um arquivo externo.

Resultado atual (`make dvc-metrics`):

| Modelo                          | `cv_pr_auc_mean` | `test_pr_auc` | `train_pr_auc` |
| ------------------------------- | ---------------- | ------------- | -------------- |
| Regressão logística             | 0,3761 ± 0,0037  | 0,3715        | 0,3762         |
| Random Forest padrão (anterior) | 0,3723 ± 0,0061  | 0,3716        | 0,4956         |
| Random Forest com tuning        | 0,4041 ± 0,0068  | 0,4022        | 0,4227         |

**O número honesto do modelo com tuning é o `test_pr_auc` de 0,4022.** O `cv_pr_auc_mean`
dele (0,4041) é o melhor de 30 candidatos avaliados nesses mesmos folds, então é um pouco
otimista; o holdout de teste a busca nunca viu. Mesmo por ele, o Random Forest com tuning
lidera com folga (0,4022 contra 0,3715 da regressão logística), e `registry.py` passa a
promover ele a `champion`. A distância entre treino e teste caiu de 0,12 para 0,02, então
o overfitting do modelo padrão praticamente sumiu. No threshold de 0,5, o recall sobe para
0,95 com precisão de 0,25 (regressão logística: 0,89 e 0,28).

Os três melhores candidatos da busca empatam (0,4041, 0,4039 e 0,4037 de CV): a escolha
entre eles é ruído. O que importa é o que eles têm em comum, `min_samples_leaf=10` e
`max_features=0.3`; `n_estimators` e `max_depth` quase não mudam o resultado.

A linha "Random Forest padrão" vem da run anterior ao tuning. Para reproduzi-la, treine
com `build_random_forest(params={})`, que usa os padrões do scikit-learn no lugar de
`TUNED_PARAMS`.

A URI e as credenciais do MLflow vêm só do ambiente (`MLFLOW_TRACKING_URI`,
`MLFLOW_TRACKING_USERNAME` e `MLFLOW_TRACKING_PASSWORD`, que podem ficar no `.env`).
Para rodar sem enviar nada ao servidor, sobrescreva a URI só naquele comando (variável
já definida no shell tem prioridade sobre o `.env`):

```bash
MLFLOW_TRACKING_URI=sqlite:///mlflow.db uv run python -m src.training.baseline
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

O `mlflow.db` e a pasta `mlruns/` ficam na raiz e não vão para o git. Sem
`MLFLOW_TRACKING_URI` em lugar nenhum, o MLflow já usa `sqlite:///mlflow.db`.

## Docker

A imagem instala só as dependências de produção (`uv sync --locked --no-dev`), roda como
usuário sem privilégios e executa o ETL por padrão. O `.env` nunca entra na imagem
(`.dockerignore`).

```bash
docker build -t purchase-propensity .

# ETL: baixa o dataset e grava data/processed/customer_propensity.parquet no host
docker run --rm -v "$(pwd)/data:/app/data" purchase-propensity

# ETL + treino do baseline, tudo dentro do container
docker run --rm purchase-propensity sh -c \
  "python -m src.data.pipeline && python -m src.training.baseline"
```

No PowerShell, troque `$(pwd)` por `${PWD}`. O build usa BuildKit (padrão no Docker
Desktop; em Docker antigo no Linux, `DOCKER_BUILDKIT=1`).

- **Dados:** sem `-v`, o que o container gera some quando ele termina. Com
  `-v "$(pwd)/data:/app/data"`, `data/raw` e `data/processed` ficam no host e o
  download do Kaggle não se repete.
- **MLflow:** sem `MLFLOW_TRACKING_URI`, a run vai para um `mlflow.db` dentro do
  container (some ao sair). Para enviar ao servidor, passe o `.env`:
  `docker run --rm --env-file .env purchase-propensity python -m src.training.baseline`.
- **Tags da run:** dentro do container não há git, então `git_sha` fica `unknown` e
  `autor` vem de `USER` (padrão `app`; sobrescreva com `-e USER=seu-nome`).
- **Linux com `--user`:** se o bind mount der permissão negada, rode com
  `--user "$(id -u):$(id -g)"`.

## Fluxo de contribuição

Duas branches longas: `develop` (integração) e `main` (estável).

- A `main` nunca recebe commit ou push direto. Código só entra nela por Pull Request
  vindo da `develop`, com o CI verde.
- No dia a dia, crie uma branch `<tipo>/<descrição-curta>` a partir da
  `develop` (mesmos tipos dos commits, ex.: `eda/taxa-por-dispositivo`,
  `exp/random-forest`) e integre nela via PR.
- Rode `make setup` antes de começar: instala os hooks que bloqueiam commit e
  push na `main`.

```
eda/funil ─────────┐
data/dvc-pipeline ─┤
exp/random-forest ─┼──► develop ──(PR + CI verde)──► main
docs/model-card ───┘
```

### Gate de branches

Repo privado numa org free: o GitHub não oferece branch protection/rulesets. O gate
tem duas camadas, e nenhuma bloqueia de fato no servidor:

- Hooks locais (`make setup`): `no-commit-to-branch` bloqueia commit na `main`;
  `scripts/reject-push-to-main.sh` (em `.git/hooks/pre-push`) bloqueia push na `main`.
  Ambos puláveis com `--no-verify`.
- CI (`.github/workflows/`): `ci.yml` roda lint + testes;
  `guard-pr-base-develop.yml` recusa PR para `main` que não venha da `develop`;
  `guard-pr-title.yml` recusa título de PR fora do padrão de commits;
  `guard-no-direct-push-main.yml` deixa o run vermelho, apontando o autor, quando
  algum commit entra na `main` sem PR mergeado.

No PR `develop -> main`, use "Create a merge commit" (não squash nem rebase):
squash cria na `main` um commit que não existe na `develop` e as duas divergem.

Depois do merge, `cleanup-merged-branch.yml` apaga a branch de origem do PR, nunca a
`develop` nem a `main`.

Se alguém furar a fila, reverta e reabra via PR da `develop`.

## Conventional Commits

Formato: `<tipo>(<escopo>): descrição no imperativo`.

| Tipo         | Quando usar                                  |
| ------------ | -------------------------------------------- |
| `feat`     | nova funcionalidade                          |
| `fix`      | correção de bug                            |
| `docs`     | documentação                               |
| `refactor` | mudança de código sem mudar comportamento  |
| `data`     | dados: DVC, ingestão, limpeza, preparação |
| `eda`      | análise exploratória (notebooks, achados)  |
| `exp`      | experimento de modelo (treino, tuning)       |
| `test`     | adição ou ajuste de testes                 |
| `ci`       | workflows do GitHub Actions                  |
| `build`    | build, Makefile, Docker                      |
| `chore`    | infra, config, dependências                 |

Exemplos:

```
eda: analisa taxa de pedido por dispositivo
data(dvc): versiona dataset customer propensity
exp(model): compara random forest e regressao logistica
feat(registry): promove melhor modelo para champion
```

O padrão é validado no `commit-msg` por hook (commitizen, tipos definidos no
`pyproject.toml`); tipo fora da tabela (ex.: `wip`) é recusado. O título do PR
segue o mesmo padrão, porque vira a mensagem do commit de merge.
