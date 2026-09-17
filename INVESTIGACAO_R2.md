# Investigação: R² baixo e previsões concentradas (5-15 em vez de 1-20)

## Sintoma reportado
As previsões (`Predicted Raw`) ficam concentradas numa faixa estreita
(ex.: 7-15 para Heading em ST), enquanto os valores reais (`Actual`)
variam na faixa completa do atributo (2-18).

## Diagnóstico

Testado com `ST / Heading`, dataset completo (9.665 jogadores ST):

| Métrica                          | Valor |
|-----------------------------------|-------|
| Desvio padrão do `Actual`         | 2.50  |
| Desvio padrão do `Predicted Raw`  | 1.20  |
| Correlação Actual x Predicted Raw | 0.52  |
| R² (ExtraTrees, melhor modelo)    | 0.25  |
| Low_R² / High_R² (extremos)       | negativos |

**Isso é um comportamento matemático esperado, não um bug.**

Qualquer modelo treinado para minimizar erro quadrático (MSE) — o que
Ridge, RandomForest, ExtraTrees e HistGradientBoosting fazem — converge
para "regredir à média" quando o sinal disponível é limitado. Com uma
correlação real de ~0.52 entre features e atributo, a previsão que
minimiza o erro esperado para um jogador "ambíguo" é ficar perto da
média (10-11), não apostar em um extremo.

### Prova experimental: calibrar a variância piora o modelo

Testei recalibrar as previsões para restaurar a variância real
(reescalonando `Predicted` para ter o mesmo desvio padrão do `Actual`):

| | R² | MAE |
|---|---|---|
| Sem calibração (atual) | **0.251** | **1.687** |
| Com variância recalibrada | 0.030 | 1.932 |

Forçar as previsões a cobrir a faixa completa piora R² e MAE. Ou seja,
o "encolhimento" que se observa nas previsões é o modelo se comportando
corretamente diante do sinal disponível — não uma falha de configuração.

### O que eu descartei como causa

- **Excesso de features "irrelevantes" diluindo o sinal**: testei um
  modelo com apenas 3-5 features específicas de heading (taxa de
  acerto no cabeceio, altura, cabeceios/90min) contra o pipeline atual
  (90 features genéricas). O modelo enxuto teve CV_R² **pior** (0.19
  vs 0.25) — RandomForest/ExtraTrees já lidam bem com features
  irrelevantes via seleção de splits.
- **Amostras com poucos minutos jogados gerando taxas ruidosas**: o
  dataset já vem filtrado (mínimo de 1000 minutos por jogador), então
  não há jogadores com 1-2 jogos distorcendo as taxas por 90 minutos.
- **Falta de interações entre features**: adicionar explicitamente
  `altura × cabeceios ganhos/90` rendeu ganho desprezível (CV_R² 0.269
  → 0.270) — árvores já capturam essa interação via splits sucessivos.

### Causa raiz real

O teto está no **sinal disponível nos dados de stats de jogo**.
Atributos do FM (como Heading) combinam desempenho observável (taxa de
acerto em duelos aéreos) com avaliação subjetiva do scout (técnica,
posicionamento no salto, timing) que não aparece nas estatísticas
agregadas de temporada. Uma correlação bruta de 0.52 entre entrada e
saída já é um teto matemático — nenhum algoritmo consegue prever melhor
que isso sem mais informação.

## Atualização: bug real encontrado nas features de divisão

Depois desta investigação inicial, foi identificado um segundo problema,
desta vez um bug de verdade (não comportamento esperado do modelo).

`analysis/division_features.py` implementa um target encoding
sofisticado da divisão do jogador (média da divisão, média
divisão+posição, força da divisão via Rating médio), com leave-one-out
correto para evitar vazamento — inclusive recalculado em cada fold do
cross-validation.

Só que essas features **nunca chegavam a ser usadas**. Em
`models/trainer.py`, a flag que ativa esse bloco é:

```python
use_division_features = bool(
    division_column
    and position
    and division_column in train_data.columns
)
```

E em `pipelines/attributes.py`, a chamada a `train_and_compare(...)`
passava `position=current_position` e `division_prior=division_prior`,
mas **nunca passava `division_column`**. Como o parâmetro tem
`default=None`, a condição acima sempre avaliava `False`, e o bloco de
features de divisão nunca era executado — apesar de todo o código de
`division_features.py` funcionar perfeitamente quando testado de forma
isolada.

**Correção**: adicionar `division_column="Division"` na chamada de
`train_and_compare` dentro de `pipelines/attributes.py`.

### Ganho medido (mesma seed, antes/depois)

| Atributo (posição ST) | R² sem | R² com | MAE sem | MAE com |
|---|---|---|---|---|
| Heading   | 0.251 | **0.291** | 1.687 | **1.638** |
| Passing   | 0.498 | **0.528** | 1.422 | **1.368** |

Ganho consistente de ~0.03-0.04 em R² e redução real de MAE em ambos
os atributos testados. Faz sentido teoricamente: a divisão em que o
jogador atua (Premier League vs. uma liga regional, por exemplo) é um
proxy real do nível geral do elenco — times melhores recrutam jogadores
com atributos mais altos em média — e esse sinal não está presente nas
estatísticas de jogo isoladas.

## Nota de acompanhamento: shrinkage agora visível direto no CSV

Depois desta investigação, `evaluate_predictions` passou a calcular
`Bias` (erro médio COM sinal) e `evaluate_by_attribute_range` passou a
reportar `Low_Bias` / `Mid_Bias` / `High_Bias`. No experimento
RM/Heading, por exemplo:

| Faixa | Bias | MAE |
|---|---|---|
| Low  | +2.34 | 2.34 |
| Mid  | -0.37 | 0.93 |
| High | -2.25 | 2.33 |

Isso é exatamente a assinatura do shrinkage documentada acima: o
modelo superestima jogadores fracos (`Low_Bias` positivo) e subestima
os fortes (`High_Bias` negativo), sem precisar rodar nenhum script
manual — a informação já sai pronta em `ATTRIBUTE_MODELS_METRICS.csv`.

## Nota de acompanhamento: LM/RM removidos, absorvidos em LW/RW

`LM` e `RM` (meio-campo lateral clássico) foram removidos da lista de
posições do experimento — não são usados como posição própria no jogo.
Jogadores que antes caíam em `M (R)`/`M (L)` agora são mapeados
diretamente para `RW`/`LW`, junto com `AM (R)`/`AM (L)`.

Isso resolve, de quebra, o problema identificado na análise do
experimento completo: LM (430 amostras) e RM (522 amostras) eram as
duas posições com pior R² (~0.25), provavelmente por falta de dados.
Após a mudança:

| Posição | Amostras antes | Amostras depois |
|---|---|---|
| RW | 5.000 | **5.340** (+340) |
| LW | 4.789 | **5.070** (+281) |

`helpers/positioning.py`: `CANONICAL_POSITIONS` não inclui mais
`LM`/`RM`, e `to_canonical_position` mapeia `M(R)→RW` / `M(L)→LW`
(antes era `M(R)→RM` / `M(L)→LM`).

## Recomendações práticas

0. **Já corrigido**: passar `division_column="Division"` em
   `pipelines/attributes.py` — é o que ativa as features de divisão que
   já estavam implementadas e testadas em `division_features.py`, mas
   nunca chegavam a ser usadas pelo modelo.
1. **Não tentar "destravar" a variância artificialmente** — isso piora
   MAE e R², como demonstrado acima. O comportamento atual está correto.
2. **Reportar `Within2`/`Within3` como métrica principal ao usuário
   final**, não R². Para Heading (já com a correção do item 0): 78% das
   previsões erram por no máximo 2 pontos, 91% por no máximo 3 — isso é
   genuinamente útil para estimar um atributo desconhecido, mesmo com
   R² "baixo".
3. Se quiser subir o teto de sinal, os caminhos mais promissores não
   testados aqui são:
   - Agregações multi-temporada do mesmo jogador (reduz ruído de uma
     única temporada).
   - Dados de scouting complementares (se disponíveis no export do FM),
     não só stats de jogo.
   - Comparar atributos entre si: alguns provavelmente têm sinal mais
     forte que Heading (ex. Pace/Aceleração, mais objetivos e menos
     dependentes de julgamento subjetivo).
