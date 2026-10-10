# Purchase Propensity

Projeto da Fase 2 da Pós Tech de Machine Learning Engineering - FIAP.

Sistema preditivo para estimar a propensão de compra de um usuário de e-commerce a partir
do comportamento de navegação. O foco é a engenharia de ML: modelo clássico em
Scikit-Learn, dados e pipeline versionados com DVC, experimentos e Model Registry
no MLflow e execução em Docker.

## Estrutura

```
purchase-propensity/
├── src/
│   ├── data/         # carregamento e split
│   ├── features/     # pré-processamento
│   ├── models/       # definição dos modelos
│   ├── training/     # treino, avaliação e registro no MLflow
│   └── config.py     # paths
├── tests/            # pytest
├── data/             # dados locais (fora do git)
├── models/           # artefatos treinados (não versionados no git)
├── configs/          # configurações e hiperparâmetros
├── notebooks/        # EDA e exploração (`NN_tema.ipynb`, ex.: `01_eda.ipynb`)
├── docs/             # documentação do projeto
├── .github/          # CI
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

## Comandos (Makefile)

| Comando           | O que faz                                    |
| ----------------- | -------------------------------------------- |
| `make setup`      | Instala dependências e hooks do git          |
| `make install`    | Instala dependências a partir do lock        |
| `make lint`       | `ruff check` + checagem de formatação        |
| `make format`     | Formata e auto-corrige com ruff              |
| `make test`       | Roda os testes (pytest)                      |
| `make pre-commit` | Roda todos os hooks em todos os arquivos     |
| `make help`       | Lista todos os alvos                         |

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
`pyproject.toml`); tipo fora da tabela (ex.: `wip`) é recusado.
