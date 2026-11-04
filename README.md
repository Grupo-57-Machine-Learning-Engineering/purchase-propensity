# Purchase Propensity

Projeto da Fase 2 da Pós Tech de Machine Learning Engineering - FIAP.

Sistema preditivo para estimar a propensão de compra de um usuário de e-commerce a partir
do comportamento de navegação. O foco é a engenharia de ML: modelo clássico em
Scikit-Learn, dados e pipeline versionados com DVC, experimentos e Model Registry
no MLflow, execução em Docker e API de inferência FastAPI.

- API no ar: https://purchase-propensity-api.onrender.com/docs

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

### Documentação

- [`docs/model_card.md`](docs/model_card.md): o modelo `champion`, métricas, uso
  pretendido e limitações.
- [`docs/ml_canvas.md`](docs/ml_canvas.md): o problema e a solução no formato do ML Canvas.
- [`docs/workflow.md`](docs/workflow.md): o passo a passo do dia a dia, do setup ao
  Model Registry.

## Estrutura

```
purchase-propensity/
├── src/
│   ├── data/         # ETL: extração (Kaggle), transformação e carga em parquet
│   ├── features/     # seleção de features (tira id, alvo e colunas de vazamento)
│   ├── models/       # estimadores do scikit-learn (regressão logística, Random Forest)
│   ├── training/     # split, treino, cross-validation, MLflow, comparação e Registry
│   ├── api/          # API FastAPI que serve o champion (POST /predict, GET /health)
│   ├── pipeline.py   # pipeline de ponta a ponta (usado pelo Docker)
│   ├── config.py     # paths, leitura do .env e de configs/params.yaml
│   └── logger.py     # logger centralizado (formato e nível únicos)
├── configs/
│   └── params.yaml   # seed, split, cross-validation, threshold e hiperparâmetros
├── tests/            # pytest, espelhando a estrutura de src/
├── data/             # raw/ (versionado pelo DVC) e processed/ (gerado pelo pipeline)
├── models/           # vazia: os modelos ficam no MLflow (artefatos e Model Registry)
├── notebooks/        # EDA (`NN_tema.ipynb`, ex.: `01_eda.ipynb`)
├── docs/             # EDA, model card, ML canvas e fluxo de trabalho
├── reports/          # métricas do pipeline DVC e resultado da busca de hiperparâmetros
├── .github/          # CI (lint, testes, build Docker), guards de branch e template de PR
├── .env.example      # modelo do .env (dataset, logs e credenciais do MLflow)
├── Dockerfile        # imagem do pipeline e da API (uv, só deps de produção, sem root)
├── render.yaml       # deploy da API no Render (Blueprint)
├── .dockerignore     # contexto de build sem .env, dados nem venv
├── dvc.yaml          # pipeline DVC: etl → train_baseline / train_random_forest → promote
├── dvc.lock          # hashes de dados, código e parâmetros da última execução
├── pyproject.toml    # dependências de produção e grupo dev (uv)
├── uv.lock           # versões exatas de todas as dependências
└── Makefile          # atalhos (make help)
```

## Visão geral técnica

| Tema | Onde está |
| --- | --- |
| Clean Code: módulos curtos, type hints, docstrings | `src/`, com ruff (`N`, `ANN`, `PLR0915` com até 20 statements por função) no pre-commit e no CI |
| POO ou Funcional | Funções puras por etapa; `Experiment` e `SplitData` (dataclasses imutáveis) ligam modelo, dados e fluxo de treino |
| uv com prod/dev separados e lock | `pyproject.toml` (`dependencies` e grupo `dev`) e `uv.lock`; `uv sync --locked` instala do zero |
| `.env.example`, `.gitignore`, `.dockerignore` | Na raiz; o `.env` nunca entra no git nem na imagem |
| Dockerfile funcional | `Dockerfile`: `uv sync --locked --no-dev` e roda o pipeline inteiro; o CI constrói a imagem |
| DVC: dataset versionado e `dvc.yaml` | `data/raw/training_sample.csv.dvc` (remote no DAGsHub) e `dvc.yaml` com 4 estágios e `params` |
| Seeds fixados | `seed` em `configs/params.yaml`, usado no split, na cross-validation, nos modelos e na busca |
| Modelo clássico no Scikit-Learn | Regressão logística e Random Forest (`src/models/`) |
| MLflow Tracking | Cada run loga parâmetros, métricas, tags e o modelo (`src/training/tracking.py`) |
| Model Registry | O melhor modelo é registrado e recebe o alias `champion` (`src/training/registry.py`) |
| API de previsão | FastAPI em `src/api/` carrega o `champion` do Registry; deploy no Render (`render.yaml`) |
| Histórico de commits do grupo | Conventional Commits validados por hook; todo código entra por PR para a `develop` |

## Setup

Pré-requisitos: [uv](https://docs.astral.sh/uv/) e `make` num shell POSIX (no Windows,
use o Git Bash; o `make` no PowerShell/cmd não roda os alvos). O Python 3.11 é
provisionado pelo uv a partir do `.python-version`.

```bash
make setup        # uv sync + hooks do git
```

O `make setup` usa `uv sync`, o equivalente do `poetry install`: cria o `.venv` e instala as
versões exatas do `uv.lock`. Para conferir que o lock está em dia, `uv sync --locked`
(o CI faz isso); para só as dependências de produção, `uv sync --no-dev` (o Docker faz
isso).

As configurações da máquina (dataset, pasta de dados, nível de log e credenciais do
MLflow) ficam no `.env`. Ele é opcional: sem ele, valem os padrões do `.env.example` e o
MLflow grava localmente. Para mudar algum valor:

```bash
cp .env.example .env
```

Os parâmetros do treino (seed, tamanho do teste, folds, threshold e hiperparâmetros) ficam
em `configs/params.yaml`, que vai para o git e é rastreado pelo DVC.

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

O pipeline está em `dvc.yaml`:

```
etl ──► train_baseline ──────┐
  └───► train_random_forest ─┴──► promote
```

- `etl` gera o parquet a partir do CSV bruto.
- `train_baseline` e `train_random_forest` treinam os dois modelos, logam a run no MLflow
  e gravam as métricas em `reports/metrics/`. Cada um depende só do seu código e das suas
  chaves de `configs/params.yaml`: mudar um hiperparâmetro do Random Forest não retreina a
  regressão logística.
- `promote` registra o melhor modelo no Model Registry com o alias `champion`.

O DVC só reexecuta o estágio cujas dependências mudaram:

```bash
make dvc            # dvc repro: roda o que estiver desatualizado
make dvc-metrics    # dvc metrics show: métricas dos dois modelos lado a lado
make dvc-dag        # grafo dos estágios
```

As métricas em `reports/metrics/` vão para o git, então `uv run dvc metrics diff` e
`uv run dvc params diff` comparam o commit atual com o anterior. Quando o dataset ou o
ETL mudam, rode `make dvc`, depois `uv run dvc push` (precisa de permissão de escrita no
DAGsHub) e commite o `dvc.lock` junto com o código: sem o push, os outros recebem um
`dvc.lock` que aponta para arquivos que não estão no remote.

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
| `make pipeline`    | ETL, treino e promoção, sem DVC            |
| `make dvc`         | Reproduz o pipeline via DVC                |
| `make dvc-metrics` | Mostra as métricas dos modelos (DVC)       |
| `make dvc-dag`     | Exibe o grafo (DAG) de estágios do DVC     |
| `make tune-rf`     | Busca hiperparâmetros do Random Forest     |
| `make docker-build`| Constrói a imagem Docker                   |
| `make docker-run`  | Roda o pipeline no container               |
| `make api`         | Sobe a API em `http://127.0.0.1:8000`      |
| `make docker-api`  | Sobe a API no container                    |
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
- **Reuso:** o fluxo (split, treino, cross-validation, métricas e MLflow) fica uma vez só
  em `src/training/experiment.py`. Cada modelo é um `Experiment` com o estimador de
  `src/models/` e o nome da run, então `src/training/baseline.py` e
  `src/training/random_forest.py` têm poucas linhas.
- **Parâmetros:** seed, split, folds, threshold e hiperparâmetros vêm de
  `configs/params.yaml`, e não do código.
- **Tracking:** experiment `purchase-propensity`, com as tags `autor`, `git_sha` e
  `dataset_version` (hash do parquet).

### Busca de hiperparâmetros

```bash
make tune-rf    # random search do Random Forest (lento: 25 minutos na última execução)
```

As 20 flags binárias formam só ~4.200 combinações distintas nas 318 mil linhas de treino,
e o Random Forest padrão (árvores sem limite) decorava as raras: PR-AUC de 0,50 no treino
contra 0,37 no teste. A busca (`src/training/tuning.py`) é um random search de 30
candidatos por PR-AUC, nos mesmos 5 folds da cross-validation das runs, focado em
regularização (`min_samples_leaf`, `max_depth`, `max_features`) e no número de árvores.

O resultado vai para `reports/tuning/random_forest.json` (melhores parâmetros e os 10
melhores candidatos) e para uma run do MLflow com a tag `kind=search`, que fica fora do
ranking por não ter modelo. Os melhores parâmetros são copiados para
`models.random_forest` em `configs/params.yaml` (um teste confere que os dois batem):
assim o `make dvc` treina sempre o mesmo modelo sem rodar a busca, que é cara.

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
  "stages", que o MLflow 2.9+/3.x descontinuou). Se o `champion` já é esse modelo, não
  cria versão nova. Para carregar o modelo promovido:
  `mlflow.sklearn.load_model("models:/purchase-propensity@champion")`.
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
com `build_random_forest(params={})`, que usa os padrões do scikit-learn no lugar dos
parâmetros de `configs/params.yaml`.

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
usuário sem privilégios e executa o pipeline inteiro por padrão (`src/pipeline.py`): ETL,
treino dos dois modelos e promoção do `champion`, na ordem do `dvc.yaml`. O DVC fica fora
da imagem (é dependência de desenvolvimento). O `.env` nunca entra na imagem
(`.dockerignore`).

```bash
docker build -t purchase-propensity .          # ou: make docker-build

# Pipeline completo, com as runs indo para o DAGsHub (credenciais do .env)
docker run --rm --env-file .env -v "$(pwd)/data:/app/data" purchase-propensity

# Pipeline completo sem enviar nada: MLflow local dentro do container
docker run --rm -v "$(pwd)/data:/app/data" purchase-propensity

# Uma etapa só
docker run --rm -v "$(pwd)/data:/app/data" purchase-propensity python -m src.data.pipeline
```

No PowerShell, troque `$(pwd)` por `${PWD}`. No Git Bash, ponha `MSYS_NO_PATHCONV=1` antes
do `docker run`: sem isso, o Git Bash converte o caminho e o container recebe uma pasta
`data` vazia (o `make docker-run` já faz isso). O build usa BuildKit (padrão no Docker
Desktop; em Docker antigo no Linux, `DOCKER_BUILDKIT=1`).

- **Dados:** sem `-v`, o que o container gera some quando ele termina. Com
  `-v "$(pwd)/data:/app/data"`, `data/raw` e `data/processed` ficam no host e o
  download do Kaggle não se repete.
- **MLflow:** sem `MLFLOW_TRACKING_URI`, as runs e o Registry vão para um `mlflow.db`
  dentro do container (some ao sair). Para enviar ao DAGsHub, passe o `.env` com
  `--env-file .env`.
- **Tags da run:** dentro do container não há git, então `git_sha` fica `unknown` e
  `autor` vem de `USER` (padrão `app`; sobrescreva com `-e USER=seu-nome`).
- **Linux com `--user`:** se o bind mount der permissão negada, rode com
  `--user "$(id -u):$(id -g)"`.

## API

`src/api/` serve o modelo `champion` com FastAPI. Ao subir, a API resolve o alias
`champion` no Model Registry, carrega aquela versão e confere se o modelo foi treinado
com as mesmas 20 colunas, na mesma ordem, que ela recebe. Por isso precisa das
credenciais do MLflow no `.env` (ou no ambiente). Sem `MLFLOW_TRACKING_URI`, procura o
Registry no `mlflow.db` local (depois de um `make dvc` ou `make pipeline` local).

```bash
make api            # http://127.0.0.1:8000/docs para testar pelo navegador
make docker-api     # a mesma API dentro do container (make docker-build antes)
```

| Endpoint | O que faz |
| --- | --- |
| `GET /health` | Estado e versão do modelo carregado |
| `POST /predict` | Score de 1 a 1.000 visitas por chamada |
| `GET /docs` | Documentação interativa (Swagger) com exemplo pronto |

Cada visita leva as 20 flags de navegação como booleanos (as 3 colunas de vazamento
não entram; mandar uma delas dá erro 422) e um `user_id` opcional, devolvido na
resposta:

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "visitors": [{"user_id": "visita-1", "basket_icon_click": true, "basket_add_list": false,
    "basket_add_detail": true, "sort_by": false, "image_picker": false,
    "account_page_click": false, "promo_banner_click": false, "detail_wishlist_add": false,
    "list_size_dropdown": false, "closed_minibasket_click": false,
    "checked_returns_detail": false, "saw_sizecharts": false, "saw_delivery": false,
    "saw_account_upgrade": false, "saw_homepage": true, "device_mobile": false,
    "device_computer": true, "device_tablet": false, "returning_user": true,
    "loc_uk": true}]
}'
```

A resposta traz, para cada visita, o `score` (0 a 1) e `purchase` (`score >= threshold`,
com o `threshold` de `configs/params.yaml`), além da versão do modelo. O score serve para
ordenar quem tem mais chance de comprar; não é probabilidade calibrada
([`docs/model_card.md`](docs/model_card.md)).

### Deploy no Render

O `render.yaml` descreve o serviço: a mesma imagem Docker, com `python -m src.api` no
lugar do pipeline, no plano gratuito. Ele publica a branch escolhida ao criar o Blueprint
(a `main`, em produção) e só republica depois que o CI passa.

1. No [Render](https://render.com), **New > Blueprint** e escolha este repositório (o
   Render precisa de acesso à organização no GitHub).
2. Preencha `MLFLOW_TRACKING_USERNAME` e `MLFLOW_TRACKING_PASSWORD` (usuário e token do
   DAGsHub). Eles ficam só no painel do Render, nunca no repositório.
3. Quando o deploy terminar, abra `https://<serviço>.onrender.com/docs`. O serviço do
   projeto está em https://purchase-propensity-api.onrender.com/docs.

No plano gratuito o serviço dorme após uns 15 minutos sem uso: a primeira chamada
depois disso demora cerca de um minuto, o tempo de subir o container e baixar o modelo
do DAGsHub.

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
