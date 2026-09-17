# FM Genial

Experimentos de Machine Learning para prever **atributos ocultos do Football
Manager** (Finishing, Heading, Passing, Marking...) a partir de estatísticas
observáveis de jogo (gols, xG, passes, desarmes, cabeceios etc.).

A ideia: no FM, os atributos "reais" de um jogador só são totalmente
conhecidos pelo motor do jogo — o scouting do usuário só enxerga estimativas
com incerteza. Este projeto treina modelos de regressão para estimar esses
atributos a partir de dados 100% observáveis (o que o jogador *fez* em
campo), posição por posição.

## Como funciona

```
Exports do FM (ATT_*.csv / STATS_*.csv)
        │
        ▼
scripts/merge_exports.py   → databases/{attributes,stats}/ALL.csv
        │
        ▼
main.py (--mode attributes)
        │
        ├─ helpers/loader.py          → limpeza (dinheiro, altura, posições)
        ├─ analysis/feature_engineering.py → métricas /90, taxas, derivadas
        ├─ analysis/division_features.py   → target encoding da divisão (leave-one-out)
        ├─ analysis/feature_selection.py   → seleção por correlação + mutual info
        ├─ models/trainer.py               → treino, CV, comparação de modelos
        └─ models/evaluator.py             → métricas (R², MAE, Within1/2/3...)
        │
        ▼
outputs/ATTRIBUTE_MODELS_METRICS_*.csv
outputs/ATTRIBUTE_FEATURE_IMPORTANCE_*.csv
outputs/ATTRIBUTE_PREDICTIONS_ALL_*.csv
```

Cada combinação **posição × atributo** (ex.: `ST` / `Finishing`) treina e
compara 4 modelos (Ridge, RandomForest, ExtraTrees, HistGradientBoosting)
contra uma baseline (`DummyRegressor`), escolhe o melhor pelo R² de
validação cruzada e só então avalia no conjunto de teste — o teste nunca
participa da escolha do modelo.

## Estrutura do projeto

| Pasta / arquivo | O que é |
|---|---|
| `main.py` | Entrypoint principal: carrega dados, roda o experimento de ML, exporta os CSVs de resultado. |
| `paths.py`, `config/settings.py` | Caminhos do projeto e hiperparâmetros padrão (test size, CV, alpha etc.). |
| `helpers/` | Parsing de dados brutos do FM (dinheiro, altura, posições), resolução de caminhos, exportação de CSV. |
| `analysis/` | Engenharia de features, seleção de features, encoding de divisão, categorização de atributos. |
| `models/` | Modelos candidatos, treino/comparação, avaliação, tuning de hiperparâmetros. |
| `pipelines/` | Lógica de mais alto nível: experimento de atributos, geração de stats limpos, merge de exports do FM. |
| `scripts/` | CLIs standalone para tarefas fora do experimento principal (merge de exports, geração de stats limpos). |
| `databases/` | Dados de entrada: `stats/` (desempenho em campo) e `attributes/` (atributos-alvo), por temporada. |
| `outputs/` | Resultados gerados (não versionado — ver `.gitignore`). |

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Dependências: `pandas`, `scikit-learn`, `joblib`, `matplotlib`.

## Preparando os dados

Os dados de entrada são exports em CSV do **FM26PlayerExport by vinteset**,
separados por `;` e codificados em `utf-8-sig`. Cada temporada precisa de um
par de arquivos com o mesmo nome em `databases/stats/` e
`databases/attributes/` (ex.: `SEASON_01.csv` nos dois).

Se você exporta por posição (`ATT_ST.csv`, `STATS_ST.csv`, `ATT_ST2.csv`...),
use o script de merge para juntar tudo num único CSV por tipo:

```bash
python scripts/merge_exports.py --exports-dir "C:\caminho\para\Exports CSV"
```

Sem `--exports-dir`, ele usa o diretório padrão do exportador (ver
`pipelines/merge_exports.py`). O merge deduplica por `Unique ID`, mantendo a
primeira ocorrência em ordem alfabética dos arquivos — pense nisso se dois
arquivos tiverem o mesmo jogador com dados diferentes.

Resultado: `databases/attributes/ALL.csv` e `databases/stats/ALL.csv`,
detectados automaticamente pelo `main.py` quando nenhuma temporada é
encontrada via `SEASON_*.csv`.

No Windows, um `.bat` simples cobre o dia a dia:

```bat
@echo off
python scripts\merge_exports.py --exports-dir "C:\caminho\para\Exports CSV"
pause
```

## Rodando o experimento

Todas as posições e atributos, multi-temporada (detecta `SEASON_*.csv`
automaticamente):

```bash
python main.py
```

Uma posição e um atributo específicos (mais rápido, bom para depuração):

```bash
python main.py --position ST --attribute Finishing -v
```

Múltiplos atributos de uma vez:

```bash
python main.py --position ST --attribute "Finishing,Heading,Long Shots"
```

Temporadas específicas ou um arquivo único:

```bash
python main.py --seasons SEASON_01,SEASON_02
python main.py --file databases/stats/SEASON_01.csv
```

Também é possível prever `Rating` diretamente (Ridge simples, sem o
pipeline completo de atributos):

```bash
python main.py --mode rating
```

### Principais flags

| Flag | Padrão | Descrição |
|---|---|---|
| `--position` | todas | Posição única (`GK`, `LB`, `CB`, `RB`, `DM`, `CM`, `LW`, `AM`, `RW`, `ST`). |
| `--attribute` / `--target` | todos | Atributo(s) específico(s), separados por vírgula. |
| `--cv` | 5 | Número de folds da validação cruzada. |
| `--min-samples` | 30 | Amostras mínimas por posição para treinar. |
| `--test-size` | 0.20 | Fração reservada para teste final. |
| `--division-prior` | 20.0 | Força do shrinkage das médias por divisão (target encoding). |
| `--correlation-threshold` | 0.95 | Limite de correlação para descartar features redundantes. |
| `--tune` | desligado | Ativa `RandomizedSearchCV` (mais lento, melhora hiperparâmetros). |
| `--save-clean-stats` | desligado | Também exporta `STATS_CLEAN.csv` (stats + features derivadas, sem treino). |
| `-q` / `-v` | — | Silencia o progresso / mostra logs de debug. |

Lista completa: `python main.py --help`.

### Gerando só o dataset limpo (sem treinar nada)

Útil para inspecionar as features derivadas (`/90`, taxas de conversão,
métricas compostas) sem rodar o experimento completo:

```bash
python scripts/clean_stats.py
```

Aceita as mesmas flags `--file` / `--seasons` do `main.py`. Gera
`outputs/STATS_CLEAN.csv`.

## Interpretando os resultados

`outputs/ATTRIBUTE_MODELS_METRICS_*.csv` traz, por combinação
posição×atributo, o melhor modelo e um conjunto amplo de métricas. Duas
merecem atenção especial:

- **R² moderado (0.25–0.45) é esperado, não um bug.** Atributos do FM
  combinam desempenho observável com avaliação subjetiva do scout (técnica,
  posicionamento, timing) que não aparece nas estatísticas agregadas de
  temporada — isso limita o teto de sinal disponível, independente do
  modelo usado.
- **`Within1` / `Within2` / `Within3`** (fração de previsões que erram por
  no máximo 1/2/3 pontos, numa escala de 20) costumam ser mais úteis na
  prática do que o R² isoladamente: é comum ver 75–85% das previsões dentro
  de 2 pontos mesmo com R² moderado.
- **`Low_Bias` / `Mid_Bias` / `High_Bias`** expõem a assinatura clássica de
  regressão à média: o modelo tende a superestimar jogadores fracos
  (`Low_Bias` positivo) e subestimar os fortes (`High_Bias` negativo).
  Tentar "destravar" essa variância artificialmente piora R² e MAE — não é
  recomendado.

`ATTRIBUTE_FEATURE_IMPORTANCE_*.csv` traz a importância por permutação de
cada feature usada pelo melhor modelo. `ATTRIBUTE_PREDICTIONS_ALL_*.csv`
traz a previsão individual por jogador, incluindo `Absolute Error` e (com
múltiplas temporadas) `Sample ID` no formato `Unique ID:Season`.

## Limitações conhecidas

- Sem testes automatizados — mudanças na engenharia de features ou no
  encoding de divisão podem introduzir vazamento de dados silenciosamente.
- Treinar todas as posições × todos os atributos pode levar bastante tempo
  (não há paralelização entre combinações, só dentro de cada modelo via
  `n_jobs=-1`).
- O caminho padrão de exports em `pipelines/merge_exports.py`
  (`DEFAULT_EXPORTS_DIR`) é específico da máquina onde o projeto foi
  desenvolvido; use sempre `--exports-dir` em outra máquina.