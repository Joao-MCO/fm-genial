from __future__ import annotations

import pandas as pd


FREE_AGENT = "Free Agent"

DIVISION_AVERAGE = "Division Average"
DIVISION_POSITION_AVERAGE = "Division Position Average"
DIVISION_STRENGTH = "Division Strength"

DIVISION_SAMPLE_SIZE = "Division Sample Size"
DIVISION_POSITION_SAMPLE_SIZE = "Division Position Sample Size"

DIVISION_FEATURES = [
    DIVISION_AVERAGE,
    DIVISION_POSITION_AVERAGE,
    DIVISION_STRENGTH,
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