# EDA: achados e decisões para a modelagem

> Achados da análise exploratória ([`notebooks/01_eda.ipynb`](../notebooks/01_eda.ipynb))
> sobre o dataset Customer Propensity to Purchase. Base:
> `data/processed/customer_propensity.parquet` (455.401 usuários, 23 flags e o alvo
> `ordered`). A EDA não treina modelos. Para reproduzir: `make etl && make eda`.
>
> Cada seção abaixo indica a seção do notebook com a evidência.

## Grupos de features (seção 2)

As 23 flags foram agrupadas pela etapa da visita em que acontecem. Essa classificação é a
base da análise de vazamento:

| Grupo | Flags |
| --- | --- |
| Fim do funil | `sign_in`, `saw_checkout`, `checked_delivery_detail` |
| Carrinho | `basket_icon_click`, `basket_add_list`, `basket_add_detail`, `closed_minibasket_click` |
| Navegação | `saw_homepage`, `list_size_dropdown`, `sort_by`, `image_picker`, `promo_banner_click`, `detail_wishlist_add`, `account_page_click`, `checked_returns_detail`, `saw_delivery`, `saw_sizecharts`, `saw_account_upgrade` |
| Contexto | `device_mobile`, `device_computer`, `device_tablet`, `returning_user`, `loc_uk` |

O Kaggle documenta só 4 colunas; as descrições das demais vêm da interpretação dos nomes.

## Qualidade e estrutura (seção 3)

- Sem nulos, sem `user_id` repetido, todas as flags booleanas (0/1 validado no ETL).
- Só 9.086 combinações distintas de flags. As 10 mais comuns cobrem 61,4% dos usuários,
  e 4.760 combinações têm um único usuário. Linhas com a mesma combinação são usuários
  diferentes, não duplicatas: nenhuma linha deve ser removida.
- Dispositivos não são exclusivos: 1.477 usuários usaram dois ou três no mesmo dia.

## Alvo (seção 4)

4,19% dos usuários fizeram pedido (19.093 de 455.401). Responder sempre "não" acerta
95,8%, então acurácia não serve como métrica, e na curva precisão x recall a linha de
base é 0,042.

## Frequência das flags (seção 5)

A distribuição é muito assimétrica: `loc_uk` (93%) e `device_mobile` (68%) aparecem em
quase todos os usuários, enquanto 7 flags ficam abaixo de 1%. A menor é `saw_sizecharts`,
com 177 casos; as taxas dessas flags raras têm intervalos de confiança largos.

## Variáveis mais associadas ao pedido (seção 6)

Fora as três do fim do funil (vazamento, seção 8), ordenadas pelo lift (taxa de pedido
com a flag dividida pela taxa geral):

| Feature | Pedido com a flag | Pedido sem a flag | Lift |
| --- | --- | --- | --- |
| `basket_icon_click` | 30,1% | 1,3% | 7,2x |
| `basket_add_detail` | 27,5% | 1,2% | 6,6x |
| `closed_minibasket_click` | 25,4% | 3,8% | 6,0x |
| `basket_add_list` | 24,5% | 2,6% | 5,8x |
| `account_page_click` | 23,4% | 4,1% | 5,6x (só 1.626 usuários) |
| `list_size_dropdown` | 9,9% | 2,5% | 2,4x |
| `saw_homepage` | 9,1% | 2,2% | 2,2x |
| `device_computer` | 6,2% | 3,7% | 1,5x |
| `returning_user` | 5,3% | 2,9% | 1,3x |
| `loc_uk` | 4,4% | 1,8% | 1,0x |
| `device_mobile` | 3,6% | 5,5% | 0,9x |

O carrinho é o melhor sinal de intenção. Navegação tem lifts altos em flags raras, mas
pesa pouco no total. `device_mobile` é o único fator que reduz a chance de pedido.

## Funil e condições necessárias (seção 7)

| Etapa | % dos visitantes |
| --- | --- |
| Interagiu com o carrinho | 17,1% |
| Fez login | 8,9% |
| Viu o checkout | 8,0% |
| Conferiu a entrega | 6,3% |
| Fez pedido | 4,2% |

- Nenhum pedido aconteceu sem `saw_checkout`: ver o checkout é condição necessária do
  pedido. Login e conferência da entrega ficam perto disso (menos de 1,1% dos pedidos sem
  eles).
- 3,9% dos pedidos não tiveram nenhuma interação com o carrinho: o carrinho é um sinal
  forte, mas não uma regra.
- Quase metade de quem vê o checkout não compra (17.378 usuários).

## Vazamento: não usar como feature (seção 8)

`sign_in`, `saw_checkout` e `checked_delivery_detail` descrevem a compra acontecendo, não
a intenção antes dela:

| Feature | Pedido com a flag | Pedido sem a flag | Pedidos que têm a flag |
| --- | --- | --- | --- |
| `saw_checkout` | 52,4% | 0,00% | 100,0% |
| `checked_delivery_detail` | 66,0% | 0,05% | 99,0% |
| `sign_in` | 46,9% | 0,03% | 99,4% |

O erro irredutível (o mínimo que qualquer regra erraria vendo só aquelas colunas) mostra
o quanto essas flags entregam o desfecho:

| Features | Combinações distintas | Erro irredutível |
| --- | --- | --- |
| Todas as 23 | 9.086 | 0,6% |
| Sem `saw_checkout` e `checked_delivery_detail` | 6.638 | 2,5% |
| Sem as 3 do fim do funil | 4.948 | 3,9% |
| Responder sempre "não" | n/a | 4,19% |

Com as 23 features o pedido fica quase determinístico, o que é sinal de vazamento. Sem
elas, o erro mínimo (3,9%) fica perto do de responder sempre "não": nenhum padrão de
intenção garante a compra. O valor do modelo está em ordenar os usuários por
probabilidade, não em classificar.

Vazamento não se resolve trocando de algoritmo: o problema é a feature não existir no
momento da previsão (antes da compra), seja qual for o modelo.

## Correlação entre features (seção 9)

As três flags do fim do funil são fortemente correlacionadas (`sign_in` x
`saw_checkout`: 0,93). Os dispositivos têm correlação negativa forte (`device_mobile` x
`device_computer`: -0,71): somam ~1 e ficam quase colineares, o que não quebra modelo
regularizado, mas pede uma categoria de referência para interpretar coeficientes. Não há
outra redundância que justifique remover features.

## Interações e segmentos (seção 10)

- Carrinho x recorrência: sem carrinho, a taxa é quase zero (0,02% dos novos, 0,4% dos
  recorrentes). Com carrinho, novos e recorrentes convertem quase igual (22,8% e 23,9%). A
  recorrência só diferencia quem ainda não chegou ao carrinho.
- Dispositivo x recorrência: os efeitos se somam. Recorrente no computador converte
  8,1%; novo no celular, 2,3%.
- Home x recorrência: de 1,4% (novo, sem ver a home) a 10,5% (recorrente que viu a home).
- Localização: fora do Reino Unido (7% dos usuários), 1,8%; no Reino Unido, 4,4%.

## Intensidade de navegação (seção 11)

Número de comportamentos de carrinho e navegação na visita (15 flags, sem fim do funil e
sem contexto):

| Comportamentos | Usuários | Taxa de pedido |
| --- | --- | --- |
| 0 | 241.034 | 0,2% |
| 1 | 110.838 | 1,8% |
| 2 | 49.765 | 6,7% |
| 3 | 24.124 | 16,3% |
| 4 | 15.299 | 26,0% |
| 5 ou mais | 14.341 | 37,8% |

A taxa cresce de forma monotônica: boa candidata a feature derivada.

## Data readiness (seção 12)

| Checagem | Resultado | Ação |
| --- | --- | --- |
| Nulos, ids duplicados, tipos | Nenhum problema | Nenhuma |
| Linhas repetidas | Usuários distintos com o mesmo padrão | Não remover |
| Dispositivos | Não exclusivos (1.477 usuários com mais de um) | Manter as 3 flags |
| Flags raras (< 1%) | 7 flags, a menor com 177 casos | Manter |
| Vazamento | `sign_in`, `saw_checkout`, `checked_delivery_detail` | Fora do cenário principal |
| Escala | Todas as features são booleanas | Sem normalização nem encoding |
| Alvo | 4,19% de positivos | Split e validação estratificados |

## Como aplicar na modelagem (seções 13 e 14)

- Features do modelo (20): `basket_icon_click`, `basket_add_list`,
  `basket_add_detail`, `closed_minibasket_click`, `saw_homepage`, `list_size_dropdown`,
  `sort_by`, `image_picker`, `promo_banner_click`, `detail_wishlist_add`,
  `account_page_click`, `checked_returns_detail`, `saw_delivery`, `saw_sizecharts`,
  `saw_account_upgrade`, `device_mobile`, `device_computer`, `device_tablet`,
  `returning_user`, `loc_uk`.
- Excluídas por vazamento (3): `sign_in`, `saw_checkout`, `checked_delivery_detail`.
  Um cenário com todas as 23 pode ser registrado só como referência do efeito do vazamento.
- Derivadas candidatas: `n_comportamentos` (soma das 15 flags de carrinho e
  navegação) e `interagiu_com_carrinho` (qualquer uma de `basket_add_list`,
  `basket_add_detail`, `basket_icon_click`).
- Interação relevante: carrinho x contexto. Modelos de árvore capturam sozinhos; modelo
  linear precisa de termo de interação.
- Pré-processamento: nenhum escalonamento, encoding ou remoção de linhas.
- Validação: split e validação cruzada estratificados, com seed fixa. O mesmo padrão
  de flags vai aparecer no treino e no teste; é esperado (usuários diferentes), não é
  vazamento.
- Métricas: PR-AUC, precision e recall no top k%, lift. Não usar acurácia.
- Desbalanceamento: pesos de classe ou ajuste do threshold pelo orçamento da campanha.

A seção 14 do notebook define essas listas como código (`MODEL_FEATURES`,
`LEAKY_FEATURES`, `DERIVED_CANDIDATES`), com checagens; o treino deve partir delas.

## Limitações

Loja fictícia, um único dia de visitas (sem sazonalidade nem validação temporal), só
flags binárias (sem tempo de sessão nem valor do carrinho). As descrições das colunas
vêm da interpretação dos nomes: o Kaggle documenta só 4 delas.
