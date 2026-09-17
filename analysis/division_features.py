from __future__ import annotations

import pandas as pd


FREE_AGENT = "Free Agent"

DIVISION_AVERAGE = "Division Average"
DIVISION_POSITION_AVERAGE = "Division Position Average"
DIVISION_STRENGTH = "Division Strength"
DIVISION_POSITION_TOP5_AVERAGE = "Division Position Top5 Average"
DIVISION_POSITION_BOTTOM5_AVERAGE = "Division Position Bottom5 Average"

DIVISION_SAMPLE_SIZE = "Division Sample Size"
DIVISION_POSITION_SAMPLE_SIZE = "Division Position Sample Size"

DIVISION_FEATURES = [
    DIVISION_AVERAGE,
    DIVISION_POSITION_AVERAGE,
    DIVISION_STRENGTH,
    DIVISION_POSITION_TOP5_AVERAGE,
    DIVISION_POSITION_BOTTOM5_AVERAGE,
    DIVISION_SAMPLE_SIZE,
    DIVISION_POSITION_SAMPLE_SIZE,
]


def normalize_division(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip()

    return values.mask(
        values.isna()
        | values.isin([
            "",
            "-",
            "--",
            "nan",
            "None",
        ]),
        FREE_AGENT,
    )


def _global_mean(
    series: pd.Series,
    fallback: float = 0.0,
) -> float:

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    return (
        float(values.mean())
        if not values.empty
        else fallback
    )


def _smoothed(
    mean: pd.Series,
    count: pd.Series,
    global_mean: float,
    prior_strength: float,
) -> pd.Series:

    mean = mean.fillna(global_mean)
    count = count.fillna(0.0)

    if prior_strength <= 0:
        return mean

    return (
        count * mean
        + prior_strength * global_mean
    ) / (
        count + prior_strength
    )


def _loo_extreme_mean(
    values: pd.Series,
    n: int,
    largest: bool,
) -> pd.Series:
    """Média dos n valores mais extremos (maiores ou menores) de um grupo,
    excluindo, para cada linha, o próprio valor dessa linha (leave-one-out).

    Vetorizado: ordena o grupo uma vez e ajusta apenas as linhas que fazem
    parte do top-(n+1), em vez de recalcular o top-n por linha.
    """

    values = pd.to_numeric(values, errors="coerce")
    valid = values.dropna()
    k = len(valid)

    result = pd.Series(float("nan"), index=values.index)

    if k == 0:
        return result

    order = valid.sort_values(ascending=not largest)
    ordered_vals = order.to_numpy()
    ordered_idx = order.index.to_numpy()

    limit = min(n + 1, k)
    top_vals = ordered_vals[:limit]
    top_idx = ordered_idx[:limit]
    total_top = float(top_vals.sum())

    default_topn = float(ordered_vals[:n].sum()) / min(n, k)
    result.loc[valid.index] = default_topn

    for position, idx in enumerate(top_idx):
        own_val = ordered_vals[position]
        remaining_sum = total_top - own_val
        remaining_count = limit - 1
        result.loc[idx] = (
            remaining_sum / remaining_count
            if remaining_count > 0
            else float("nan")
        )

    return result


def _extreme_mean_no_exclusion(
    values: pd.Series,
    n: int,
    largest: bool,
) -> float:
    """Média dos n valores mais extremos de um grupo, sem excluir ninguém.
    Usado para o conjunto de teste/validação, que consulta o treino como
    referência fixa e nunca precisa se auto-excluir.
    """

    valid = pd.to_numeric(values, errors="coerce").dropna()

    if valid.empty:
        return float("nan")

    order = valid.sort_values(ascending=not largest)
    top = order.to_numpy()[:n]

    return float(top.mean())



def _position_mask(
    reference: pd.DataFrame,
    position: str,
) -> pd.Series:

    if "All Positions" not in reference.columns:
        return pd.Series(
            True,
            index=reference.index,
        )

    return (
        reference["All Positions"]
        .fillna("")
        .str.split(",")
        .apply(
            lambda positions:
            position in positions
        )
    )


def add_division_features(
    train: pd.DataFrame,
    other: pd.DataFrame,
    reference_train: pd.DataFrame,
    target: str,
    *,
    position: str,
    id_column: str = "Unique ID",
    division_column: str = "Division",
    rating_column: str = "Rating",
    prior_strength: float = 20.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    train = train.copy()
    other = other.copy()
    reference = reference_train.copy()

    for frame in (
        train,
        other,
        reference,
    ):
        frame[division_column] = normalize_division(
            frame[division_column]
        )

    reference[target] = pd.to_numeric(
        reference[target],
        errors="coerce",
    )

    if rating_column in reference.columns:
        reference[rating_column] = pd.to_numeric(
            reference[rating_column],
            errors="coerce",
        )

    global_mean = _global_mean(
        reference[target],
        _global_mean(train[target]),
    )

    ref_div = (
        reference
        .groupby(division_column)[target]
        .agg(["sum", "count"])
    )

    train_target = pd.to_numeric(
        train[target],
        errors="coerce",
    )

    train_div = train[division_column]

    div_sum = train_div.map(
        ref_div["sum"]
    )

    div_count = train_div.map(
        ref_div["count"]
    ).fillna(0.0)

    if (
        id_column in train.columns
        and id_column in reference.columns
    ):
        reference_ids = set(
            reference[id_column].astype(str)
        )

        own_in_reference = (
            train_target.notna()
            & train[id_column]
            .astype(str)
            .isin(reference_ids)
        )
    else:
        own_in_reference = train_target.notna()

    # Leave-one-out
    div_sum = (
        div_sum
        - train_target
        .where(
            own_in_reference,
            0,
        )
        .fillna(0)
    )

    div_count = (
        div_count
        - own_in_reference.astype(int)
    )

    div_mean = div_sum.div(
        div_count.where(
            div_count > 0
        )
    )

    train[DIVISION_AVERAGE] = _smoothed(
        div_mean,
        div_count.clip(lower=0),
        global_mean,
        prior_strength,
    )

    train[DIVISION_SAMPLE_SIZE] = (
        div_count.clip(lower=0)
    )

    # -----------------------------------------
    # Divisão + posição
    # -----------------------------------------

    position_ref = reference.loc[
        _position_mask(
            reference,
            position,
        )
    ].copy()

    position_global = _global_mean(
        position_ref[target],
        global_mean,
    )

    pos_div = (
        position_ref
        .groupby(division_column)[target]
        .agg(["sum", "count"])
    )

    pos_sum = train_div.map(
        pos_div["sum"]
    )

    pos_count = train_div.map(
        pos_div["count"]
    ).fillna(0.0)

    own_in_position_reference = (
        own_in_reference
        & _position_mask(
            train,
            position,
        )
    )

    pos_sum = (
        pos_sum
        - train_target
        .where(
            own_in_position_reference,
            0,
        )
        .fillna(0)
    )

    pos_count = (
        pos_count
        - own_in_position_reference.astype(int)
    )

    pos_mean = pos_sum.div(
        pos_count.where(
            pos_count > 0
        )
    )

    train[DIVISION_POSITION_AVERAGE] = _smoothed(
        pos_mean,
        pos_count.clip(lower=0),
        position_global,
        prior_strength,
    )

    train[DIVISION_POSITION_SAMPLE_SIZE] = (
        pos_count.clip(lower=0)
    )

    # -----------------------------------------
    # Divisão + posição: média dos 5 melhores / 5 piores
    # -----------------------------------------
    # Usa position_ref (referência), mas o LOO é aplicado apenas
    # às linhas de `train` que também pertencem à referência —
    # mesmo critério de own_in_position_reference usado acima.

    position_ref_target = pd.to_numeric(
        position_ref[target],
        errors="coerce",
    )

    top5_by_division: dict[str, float] = {}
    bottom5_by_division: dict[str, float] = {}

    top5_result = pd.Series(float("nan"), index=train.index)
    bottom5_result = pd.Series(float("nan"), index=train.index)

    # position_ref preserva o índice original de `reference` (que, na
    # chamada real, é a mesma tabela que `train`). Isso permite localizar
    # a própria linha de cada jogador dentro do grupo sem depender de
    # Unique ID, que pode se repetir entre temporadas.
    for division_value, group in position_ref_target.groupby(
        position_ref[division_column]
    ):
        top5_by_division[division_value] = _extreme_mean_no_exclusion(
            group, 5, largest=True
        )
        bottom5_by_division[division_value] = _extreme_mean_no_exclusion(
            group, 5, largest=False
        )

        train_in_group = (
            own_in_position_reference
            & (train[division_column] == division_value)
        )

        if not train_in_group.any():
            continue

        loo_top5 = _loo_extreme_mean(group, 5, largest=True)
        loo_bottom5 = _loo_extreme_mean(group, 5, largest=False)

        # Índices de `train` que também aparecem em `group` (mesma
        # posição de linha na tabela original) recebem o valor LOO;
        # os demais (jogadores fora da referência, ex.: fold de
        # validação) ficam de fora e caem no default calculado abaixo.
        shared_idx = train.index[train_in_group].intersection(group.index)

        top5_result.loc[shared_idx] = loo_top5.loc[shared_idx]
        bottom5_result.loc[shared_idx] = loo_bottom5.loc[shared_idx]

    default_top5 = train_div.map(top5_by_division)
    default_bottom5 = train_div.map(bottom5_by_division)

    top5_result = top5_result.fillna(default_top5)
    bottom5_result = bottom5_result.fillna(default_bottom5)

    train[DIVISION_POSITION_TOP5_AVERAGE] = _smoothed(
        top5_result,
        pos_count.clip(lower=0),
        position_global,
        prior_strength,
    )

    train[DIVISION_POSITION_BOTTOM5_AVERAGE] = _smoothed(
        bottom5_result,
        pos_count.clip(lower=0),
        position_global,
        prior_strength,
    )

    # -----------------------------------------
    # Força da divisão
    # -----------------------------------------

    if rating_column in reference.columns:

        strength_global = _global_mean(
            reference[rating_column]
        )

        strength = (
            reference
            .groupby(division_column)[rating_column]
            .agg(["sum", "count"])
        )

        strength_sum = train_div.map(
            strength["sum"]
        )

        strength_count = train_div.map(
            strength["count"]
        ).fillna(0.0)

        train_rating = pd.to_numeric(
            train.get(rating_column),
            errors="coerce",
        )

        own_rating = (
            train_rating
            .where(
                own_in_reference,
                0,
            )
            .fillna(0)
        )

        own_rating_count = (
            train_rating.notna()
            & own_in_reference
        ).astype(int)

        strength_sum = (
            strength_sum
            - own_rating
        )

        strength_count = (
            strength_count
            - own_rating_count
        )

        train[DIVISION_STRENGTH] = (
            strength_sum
            .div(
                strength_count.where(
                    strength_count > 0
                )
            )
            .fillna(strength_global)
        )

    else:
        train[DIVISION_STRENGTH] = 0.0

    # -----------------------------------------
    # Transformação para validação/teste
    # -----------------------------------------

    def transform(
        frame: pd.DataFrame,
    ) -> pd.DataFrame:

        frame = frame.copy()

        frame[division_column] = normalize_division(
            frame[division_column]
        )

        div_mean = frame[
            division_column
        ].map(
            (
                ref_div["sum"]
                / ref_div["count"]
            ).replace(
                [float("inf"), -float("inf")],
                pd.NA,
            )
        )

        div_count = frame[
            division_column
        ].map(
            ref_div["count"]
        ).fillna(0.0)

        frame[DIVISION_AVERAGE] = _smoothed(
            div_mean,
            div_count,
            global_mean,
            prior_strength,
        )

        frame[DIVISION_SAMPLE_SIZE] = (
            div_count
        )

        pos_mean = frame[
            division_column
        ].map(
            (
                pos_div["sum"]
                / pos_div["count"]
            ).replace(
                [float("inf"), -float("inf")],
                pd.NA,
            )
        )

        pos_count = frame[
            division_column
        ].map(
            pos_div["count"]
        ).fillna(0.0)

        frame[
            DIVISION_POSITION_AVERAGE
        ] = _smoothed(
            pos_mean,
            pos_count,
            position_global,
            prior_strength,
        )

        top5_mean = frame[division_column].map(top5_by_division)
        bottom5_mean = frame[division_column].map(bottom5_by_division)

        frame[DIVISION_POSITION_TOP5_AVERAGE] = _smoothed(
            top5_mean,
            pos_count,
            position_global,
            prior_strength,
        )

        frame[DIVISION_POSITION_BOTTOM5_AVERAGE] = _smoothed(
            bottom5_mean,
            pos_count,
            position_global,
            prior_strength,
        )

        frame[
            DIVISION_POSITION_SAMPLE_SIZE
        ] = pos_count

        if rating_column in reference.columns:

            strength_mean = frame[
                division_column
            ].map(
                (
                    strength["sum"]
                    / strength["count"]
                ).replace(
                    [float("inf"), -float("inf")],
                    pd.NA,
                )
            )

            frame[
                DIVISION_STRENGTH
            ] = strength_mean.fillna(
                strength_global
            )

        else:
            frame[DIVISION_STRENGTH] = 0.0

        return frame

    return train, transform(other)