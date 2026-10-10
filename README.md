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
│   └── training/     # treino, avaliação e registro no MLflow
├── tests/            # pytest
├── data/             # dados locais (fora do git)
├── models/           # artefatos treinados (não versionados no git)
├── configs/          # configurações e hiperparâmetros
├── notebooks/        # EDA e exploração (`NN_tema.ipynb`, ex.: `01_eda.ipynb`)
└── docs/             # documentação do projeto
```
