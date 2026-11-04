# ML Canvas: propensão de compra

Resumo do problema e da solução no formato do Machine Learning Canvas. A loja do
dataset é fictícia, então o uso do modelo descrito aqui é uma proposta: o projeto treina,
avalia e registra o modelo, mas não o coloca em uso. Fatos sobre os dados e o modelo
estão em [`eda-findings.md`](eda-findings.md) e [`model_card.md`](model_card.md).

## Proposta de valor

Uma loja online quer saber quais visitantes têm mais chance de fazer um pedido, para
concentrar ações (oferta, contato, remarketing) em quem tem mais chance de converter. Só
4,2% dos visitantes do dataset compram, então agir sobre todos desperdiça esforço. A
própria descrição do dataset no Kaggle parte desse cenário: hoje a loja gasta com
remarketing para todos os visitantes, e o modelo serve para achar os mais promissores.

## Tarefa de ML

Classificação binária. Entrada: as 20 flags de comportamento de uma visita. Saída: um
score de propensão para `ordered` (fez pedido). O score é usado para ordenar os
usuários, não como probabilidade calibrada.

## Decisões

Proposta de uso: ordenar os visitantes pelo score e agir sobre os primeiros, num
recorte definido pelo orçamento da ação. No teste, os 10% com maior score concentram
78,5% dos pedidos (precisão de 0,329, lift de 7,9x). O projeto não define o orçamento
nem o custo da ação, então não fixa esse recorte.

## Fontes de dados

[Customer Propensity to Purchase](https://www.kaggle.com/datasets/benpowis/customer-propensity-to-purchase-data)
(Kaggle, CC0): `training_sample.csv`, com um dia de visitas de 455.401 usuários. O
arquivo é versionado com DVC e fica no remote do DAGsHub. O `testing_sample.csv` do
Kaggle não é usado porque todas as suas 151.655 linhas têm `ordered = 0`.

## Coleta de dados

O ETL (`src/data/`) baixa o dataset pelo `kagglehub`, padroniza os nomes das colunas e
converte as flags em booleanos. Num uso real, as mesmas flags precisariam ser montadas a
partir dos eventos de navegação da sessão, com os mesmos nomes e o mesmo significado.

## Features

20 flags binárias:

- carrinho: `basket_icon_click`, `basket_add_list`, `basket_add_detail`,
  `closed_minibasket_click`;
- navegação: `saw_homepage`, `list_size_dropdown`, `sort_by`, `image_picker`,
  `promo_banner_click`, `detail_wishlist_add`, `account_page_click`,
  `checked_returns_detail`, `saw_delivery`, `saw_sizecharts`, `saw_account_upgrade`;
- contexto: `device_mobile`, `device_computer`, `device_tablet`, `returning_user`,
  `loc_uk`.

`sign_in`, `saw_checkout` e `checked_delivery_detail` ficam de fora por vazamento: elas
acontecem durante a compra (nenhum pedido aconteceu sem `saw_checkout`).

## Construção do modelo

- Pipeline DVC (`dvc.yaml`): `etl`, `train_baseline`, `train_random_forest` e `promote`.
- Dois modelos do scikit-learn: regressão logística (baseline) e Random Forest com os
  hiperparâmetros de um random search de 30 candidatos.
- Split estratificado 70/30 e cross-validation de 5 folds, com seed 42
  (`configs/params.yaml`).
- Cada treino vira uma run no MLflow do DAGsHub. O `promote` registra a run com maior
  PR-AUC na cross-validation e aponta o alias `champion` para ela.

## Previsões

Hoje as previsões são feitas em lote, carregando o `champion` do Registry
(`models:/purchase-propensity@champion`) e chamando `predict_proba` sobre as flags. Não
há API nem agendamento. Como as features só ficam completas no fim da visita, o uso
natural é em lote após a sessão (por exemplo, remarketing no dia seguinte).

## Avaliação offline

| Métrica no teste | Random Forest (`champion`) | Regressão logística |
| --- | --- | --- |
| PR-AUC | 0,4022 | 0,3715 |
| ROC-AUC | 0,9513 | 0,9465 |

Um modelo aleatório fica em 0,042 de PR-AUC. Os critérios de escolha e as limitações
estão no [`model_card.md`](model_card.md).
