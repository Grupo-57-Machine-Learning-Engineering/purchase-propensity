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

### Principais achados da EDA

A análise completa está em [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb).

- Todo pedido passou pelo checkout. `saw_checkout`, `checked_delivery_detail` e `sign_in`
  acontecem junto com a compra (vazamento) e ficam fora do cenário principal de
  modelagem.
- Sem essas três features, a PR-AUC na validação cruzada fica em ~0,37-0,41, contra 0,04
  da linha de base. Com elas, chega a ~0,91, mas só reconhece quem já está comprando.
- O carrinho é o melhor sinal legítimo de intenção (lift de 6,5x a 7,2x). Desktop e
  usuário recorrente convertem mais; o recorrente no computador converte 8,1%, contra
  2,3% do novo no celular.

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
├── .github/          # CI e template de PR
├── .env.example      # modelo do .env (configurações)
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

## Comandos (Makefile)

| Comando           | O que faz                                    |
| ----------------- | -------------------------------------------- |
| `make setup`      | Instala dependências e hooks do git          |
| `make install`    | Instala dependências a partir do lock        |
| `make lint`       | `ruff check` + checagem de formatação        |
| `make format`     | Formata e auto-corrige com ruff              |
| `make test`       | Roda os testes (pytest)                      |
| `make etl`        | Baixa o dataset e gera `data/processed/`     |
| `make eda`        | Reexecuta o notebook de EDA                  |
| `make pre-commit` | Roda todos os hooks em todos os arquivos     |
| `make help`       | Lista todos os alvos                         |

## Fluxo de contribuição

Duas branches longas: `develop` (integração) e `main` (estável).

- A `main` nunca recebe commit ou push direto. Código só entra nela por Pull Request
  vindo da `develop`, com o CI verde.
- No dia a dia, crie uma branch `<tipo>/<descrição-curta>` a partir da
  `develop` (mesmos tipos dos commits, ex.: `eda/distribuicao-revenue`,
  `exp/random-forest`) e integre nela via PR.
- Rode `make setup` antes de começar: instala os hooks que bloqueiam commit e
  push na `main`.

```
eda/revenue ───────┐
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

| Tipo       | Quando usar                               |
| ---------- | ----------------------------------------- |
| `feat`     | nova funcionalidade                       |
| `fix`      | correção de bug                           |
| `docs`     | documentação                              |
| `refactor` | mudança de código sem mudar comportamento |
| `data`     | dados: DVC, ingestão, limpeza, preparação |
| `eda`      | análise exploratória (notebooks, achados) |
| `exp`      | experimento de modelo (treino, tuning)    |
| `test`     | adição ou ajuste de testes                |
| `ci`       | workflows do GitHub Actions               |
| `build`    | build, Makefile, Docker                   |
| `chore`    | infra, config, dependências               |

Exemplos:

```
eda: analisa distribuicao de Revenue por mes
data(dvc): versiona dataset online shoppers
exp(model): compara random forest e regressao logistica
feat(registry): promove melhor modelo para champion
```

O padrão é validado no `commit-msg` por hook (commitizen, tipos definidos no
`pyproject.toml`); tipo fora da tabela (ex.: `wip`) é recusado. O título do PR
segue o mesmo padrão, porque vira a mensagem do commit de merge.
