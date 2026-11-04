# Model card: purchase-propensity

Ficha do modelo promovido a `champion` no Model Registry (`purchase-propensity`). O
projeto não tem deploy: o modelo fica registrado no MLflow do DAGsHub. Os
números vêm de `reports/metrics/` (gerados pelo `dvc repro`) e das runs do MLflow no
DAGsHub. A seção [Como reproduzir](#como-reproduzir) mostra de onde sai cada um.

## Modelo

| Item | Valor |
| --- | --- |
| Tipo | `RandomForestClassifier` (scikit-learn) |
| Hiperparâmetros | `n_estimators=400`, `max_depth=12`, `min_samples_leaf=10`, `max_features=0.3`, `class_weight="balanced"` |
| Seed | 42 (`configs/params.yaml`) |
| Registry | `purchase-propensity`, versão 2, alias `champion` |
| Run no MLflow | `random-forest-tuned-no-leakage` (`4375e8709f614393b8bfce4f74dd07fa`) |
| Código | commit `990d8cc` (`develop`) |
| Como carregar | `mlflow.sklearn.load_model("models:/purchase-propensity@champion")` |

Os hiperparâmetros saíram de um random search de 30 candidatos com 5 folds
(`src/training/tuning.py`, resultado em `reports/tuning/random_forest.json`). Os três
melhores candidatos ficaram empatados (0,4041, 0,4039 e 0,4037 de PR-AUC na
cross-validation), e todos usam `min_samples_leaf=10` e `max_features=0.3`.

## Uso pretendido

Ordenar os visitantes de uma loja online pela chance de fazer um pedido, usando só o
comportamento de navegação da visita. A saída serve para priorizar quem recebe uma ação
(por exemplo, uma oferta), e não para decidir sozinha se a pessoa vai comprar.

Fora do escopo:

- Prever compra antes de a visita acontecer: as features descrevem o que a pessoa fez
  na visita.
- Usar o score como probabilidade calibrada (ver [Limitações](#limitações)).
- Aplicar em outra loja ou período sem avaliar de novo: o dataset é de um único dia de
  uma loja fictícia.

## Dados

[Customer Propensity to Purchase](https://www.kaggle.com/datasets/benpowis/customer-propensity-to-purchase-data)
(Kaggle, CC0), arquivo `training_sample.csv`, versionado com DVC.

- 455.401 usuários e 23 flags binárias. Alvo `ordered`: 19.093 pedidos (4,19%).
- Split estratificado 70/30 com seed 42: 318.780 linhas de treino (13.365 pedidos) e
  136.621 de teste (5.728 pedidos).
- 20 features. Ficam de fora `sign_in`, `saw_checkout` e `checked_delivery_detail`, que
  descrevem a compra acontecendo (nenhum pedido sem `saw_checkout`). Detalhes em
  [`eda-findings.md`](eda-findings.md), seção de vazamento.
- Sem pré-processamento: as flags entram como booleanos.

## Métricas

PR-AUC é a métrica principal, porque só 4,2% dos usuários compram e a acurácia engana
(responder sempre "não" acerta 95,8%). Na curva precisão x recall, um modelo aleatório
fica em 0,042.

| Métrica | Random Forest (`champion`) | Regressão logística (baseline) |
| --- | --- | --- |
| PR-AUC, cross-validation (5 folds) | 0,4041 ± 0,0068 | 0,3761 ± 0,0037 |
| PR-AUC, teste | 0,4022 | 0,3715 |
| PR-AUC, treino | 0,4227 | 0,3762 |
| ROC-AUC, teste | 0,9513 | 0,9465 |
| Precisão no threshold 0,5, teste | 0,2462 | 0,2811 |
| Recall no threshold 0,5, teste | 0,9532 | 0,8884 |

O `cv_pr_auc_mean` do Random Forest é um pouco otimista, porque é o melhor de 30
candidatos avaliados nos mesmos folds. O `test_pr_auc` (0,4022) vem de dados que a busca
nunca viu. A diferença entre treino e teste é de 0,02.

Ordenando o teste pelo score do `champion`:

| Recorte | Precisão | Recall | Lift |
| --- | --- | --- | --- |
| 1% com maior score | 0,518 | 0,124 | 12,4x |
| 5% com maior score | 0,411 | 0,490 | 9,8x |
| 10% com maior score | 0,329 | 0,785 | 7,9x |
| 20% com maior score | 0,203 | 0,968 | 4,8x |

Como as 20 flags formam poucas combinações distintas, muitos usuários têm o mesmo score.
Nesses empates, quem fica dentro do recorte depende da ordem das linhas, então os valores
acima podem variar um pouco.

## Limitações

- O score não é uma probabilidade calibrada. O peso de classe balanceado infla os
  valores: a média do score no teste é 0,176, contra 4,2% de compras reais. Serve para
  ordenar, não para estimar quantas pessoas vão comprar.
- No threshold de 0,5, o modelo marca 16,2% do teste como compra e acerta 1 em cada 4.
  O threshold não foi ajustado; o ponto de corte certo depende do custo da ação e do
  orçamento, que o projeto não define.
- Um único dia de visitas: não há sazonalidade nem validação temporal.
- Só flags binárias: não há tempo de sessão, valor do carrinho nem histórico de compras.
- A página do dataset no Kaggle descreve só 4 das 23 flags. O significado das outras vem
  da interpretação dos nomes.
- Sem as colunas de vazamento, o erro mínimo possível fica em 3,9% (seção de vazamento
  da EDA). O modelo não consegue separar bem compradores de não compradores. O valor está
  em ordenar.

## Considerações éticas

- O dataset não tem dado pessoal além do `user_id`, que não entra no modelo.
- `loc_uk` e as flags de dispositivo entram como features. Usuários fora do Reino Unido
  convertem menos (1,8% contra 4,4%), então tendem a receber scores menores. Se o score
  decidir quem recebe benefício, vale conferir o efeito por localização e dispositivo.

## Como reproduzir

```bash
make dvc            # treina os dois modelos e grava reports/metrics/*.json
make dvc-metrics    # tabela de métricas
```

A tabela de recortes foi calculada com o `champion` carregado do Registry:

```python
import numpy as np
import pandas as pd
import src.config  # carrega o .env com a URI e o token do MLflow
from mlflow.sklearn import load_model
from src.training.split import stratified_split

model = load_model("models:/purchase-propensity@champion")
data = stratified_split(pd.read_parquet("data/processed/customer_propensity.parquet"))
scores = model.predict_proba(data.features_test)[:, 1]
target = data.target_test.to_numpy()
order = np.argsort(-scores)
for share in (0.01, 0.05, 0.10, 0.20):
    top = order[: int(len(scores) * share)]
    precision = target[top].mean()
    print(share, precision, target[top].sum() / target.sum(), precision / target.mean())
```
