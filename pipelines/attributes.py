from __future__ import annotations

import logging
import time

import pandas as pd

from analysis.categories import (
    get_attribute_category,
)
from analysis.feature_engineering import (
    build_features,
)
from config.settings import (
    DEFAULT_ALPHA,
    DEFAULT_CORRELATION_THRESHOLD,
    DEFAULT_CV,
    DEFAULT_DIVISION_PRIOR,
    DEFAULT_MIN_SAMPLES,
    DEFAULT_PERMUTATION_REPEATS,
    DEFAULT_TEST_SIZE,
    RANDOM_STATE,
)
from helpers.positioning import (
    CANONICAL_POSITIONS,
)
from models.trainer import (
    train_and_compare,
)

logger = logging.getLogger(__name__)


def _merge(
    stats: pd.DataFrame,
    attrs: pd.DataFrame,
) -> pd.DataFrame:

    stats = stats.copy()
    attrs = attrs.copy()

    stats["_stats_index"] = stats.index

    if (
        "Unique ID" in stats.columns
        and "Unique ID" in attrs.columns
    ):

        stats["Unique ID"] = (
            stats["Unique ID"]
            .astype(str)
            .str.strip()
        )

        attrs["Unique ID"] = (
            attrs["Unique ID"]
            .astype(str)
            .str.strip()
        )

        merge_keys = ["Unique ID"]
        if "Season" in stats.columns and "Season" in attrs.columns:
            merge_keys.append("Season")

        return stats.merge(
            attrs,
            on=merge_keys,
            how="inner",
            suffixes=(
                "_stats",
                "_attributes",
            ),
            validate="one_to_one",
        )

    if (
        "Player" not in stats.columns
        or "Player" not in attrs.columns
    ):
        raise ValueError(
            "É necessário ter 'Unique ID' nos dois arquivos "
            "ou 'Player' nos dois arquivos."
        )

    # Alguns exports antigos não possuem Unique ID.
    stats["_player_key"] = (
        stats["Player"]
        .astype(str)
        .str.strip()
        .str.casefold()
    )

    attrs["_player_key"] = (
        attrs["Player"]
        .astype(str)
        .str.strip()
        .str.casefold()
    )

    stats["_player_occurrence"] = (
        stats.groupby(
            "_player_key"
        ).cumcount()
    )

    attrs["_player_occurrence"] = (
        attrs.groupby(
            "_player_key"
        ).cumcount()
    )

    merged = stats.merge(
        attrs,
        on="Unique ID",
        how="inner",
        suffixes=("_stats", "_attributes"),
        validate="one_to_one",
    )

    if "Division" not in merged.columns:

        division_stats = merged.get(
            "Division_stats"
        )

        division_attributes = merged.get(
            "Division_attributes"
        )

        if division_stats is not None:
            merged["Division"] = division_stats

        elif division_attributes is not None:
            merged["Division"] = division_attributes

    if "Unique ID" not in merged.columns:

        merged["Unique ID"] = (
            merged["_player_key"]
            + ":"
            + merged["_player_occurrence"]
            .astype(str)
        )

    return merged


def run_attribute_experiment(
    stats: pd.DataFrame,
    attributes: pd.DataFrame,
    *,
    position: str | None = None,
    attribute: str | None = None,
    cv: int = DEFAULT_CV,
    alpha: float = DEFAULT_ALPHA,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    test_size: float = DEFAULT_TEST_SIZE,
    correlation_threshold: float = DEFAULT_CORRELATION_THRESHOLD,
    tune: bool = False,
    permutation_repeats: int = DEFAULT_PERMUTATION_REPEATS,
    division_prior: float = DEFAULT_DIVISION_PRIOR,
) -> dict[str, pd.DataFrame]:

    merged = _merge(
        stats,
        attributes,
    )

    # ============================================================
    # ATRIBUTOS
    # ============================================================

    non_attribute_columns = {
        "Unique ID",
        "Player",
        "_player_key",
        "Division",
        "Season",
    }

    attribute_columns = [
        c
        for c in attributes.columns
        if c not in non_attribute_columns
    ]

    if attribute:
        matches = {
            c.casefold(): c
            for c in attribute_columns
        }

        requested_attributes = [
            item.strip()
            for item in attribute.split(",")
            if item.strip()
        ]

        invalid_attributes = [
            item
            for item in requested_attributes
            if item.casefold() not in matches
        ]

        if invalid_attributes:
            raise ValueError(
                "Atributo(s) inválido(s): "
                + ", ".join(invalid_attributes)
                + "\nOpções disponíveis: "
                + ", ".join(attribute_columns)
            )

        attribute_columns = [
            matches[item.casefold()]
            for item in requested_attributes
        ]

    # ============================================================
    # POSIÇÕES
    # ============================================================

    positions = (
        [position.upper()]
        if position
        else list(CANONICAL_POSITIONS)
    )

    if any(
        p not in CANONICAL_POSITIONS
        for p in positions
    ):
        raise ValueError(
            "Posição inválida. Opções: "
            + ", ".join(
                CANONICAL_POSITIONS
            )
        )

    stats_only = stats.copy()

    metric_rows = []
    importance_rows = []
    prediction_frames = []

    total_combinations = len(positions) * len(attribute_columns)
    logger.info(
        "Experimento iniciado: %d posição(ões) x %d atributo(s) = "
        "até %d combinações (algumas podem ser puladas por min_samples).",
        len(positions), len(attribute_columns), total_combinations,
    )

    experiment_start = time.perf_counter()
    combinations_done = 0
    combinations_skipped = 0

    # ============================================================
    # POSIÇÕES
    # ============================================================

    for position_idx, current_position in enumerate(positions, start=1):

        if "All Positions" in merged.columns:

            mask = (
                merged["All Positions"]
                .fillna("")
                .str.split(",")
                .apply(
                    lambda ps:
                    current_position in ps
                )
            )

        else:

            mask = pd.Series(
                True,
                index=merged.index,
            )

        position_df = merged.loc[
            mask
        ].copy()

        if len(position_df) < min_samples:
            combinations_skipped += len(attribute_columns)
            logger.info(
                "[%d/%d] Posição %s: apenas %d amostras (mínimo %d) — pulando.",
                position_idx, len(positions), current_position,
                len(position_df), min_samples,
            )
            continue

        logger.info(
            "[%d/%d] Posição %s: %d amostras, treinando %d atributo(s)...",
            position_idx, len(positions), current_position,
            len(position_df), len(attribute_columns),
        )

        # ========================================================
        # FEATURES
        # ========================================================

        stats_indices = (
            position_df["_stats_index"]
            .astype(int)
            .tolist()
        )

        X_features = build_features(
            stats_only.loc[
                stats_indices
            ]
        ).copy()

        X_features.index = (
            position_df.index
        )

        position_model_df = pd.concat(
            [
                position_df.drop(
                    columns=X_features.columns,
                    errors="ignore",
                ),
                X_features,
            ],
            axis=1,
        )

        # ========================================================
        # DIVISION
        # ========================================================

        if (
            "Division_attributes"
            in position_model_df.columns
        ):

            position_model_df["Division"] = (
                position_model_df[
                    "Division_attributes"
                ]
            )

        elif (
            "Division_stats"
            in position_model_df.columns
        ):

            position_model_df["Division"] = (
                position_model_df[
                    "Division_stats"
                ]
            )

        elif "Division" not in position_model_df.columns:

            raise ValueError(
                "Coluna Division não encontrada "
                "nos dados mesclados."
            )

        valid_features = list(
            X_features.columns
        )

        # ========================================================
        # ATRIBUTOS
        # ========================================================

        for attr_idx, target in enumerate(attribute_columns, start=1):

            attr_start = time.perf_counter()

            try:
                result = train_and_compare(
                    position_model_df.assign(
                        **{
                            target: pd.to_numeric(
                                position_model_df[target],
                                errors="coerce",
                            )
                        }
                    ),
                    valid_features,
                    target,
                    test_size=test_size,
                    cv=cv,
                    alpha=alpha,
                    correlation_threshold=correlation_threshold,
                    tune=tune,
                    random_state=RANDOM_STATE,
                    permutation_repeats=permutation_repeats,

                    position=current_position,
                    division_prior=division_prior,
                    division_column="Division",
                )
            except ValueError as exc:
                combinations_skipped += 1
                logger.warning(
                    "  [%d/%d] %s / %s: pulado (%s)",
                    attr_idx, len(attribute_columns), current_position,
                    target, exc,
                )
                continue

            combinations_done += 1
            attr_elapsed = time.perf_counter() - attr_start

            # ====================================================
            # MÉTRICAS
            # ====================================================

            metrics = (
                result["metrics"]
                .copy()
            )

            metrics.insert(
                0,
                "Position",
                current_position,
            )

            metrics.insert(
                1,
                "Attribute",
                target,
            )

            metrics.insert(
                2,
                "Attribute Category",
                get_attribute_category(
                    target
                ),
            )

            metrics["Samples"] = (
                len(position_df)
            )

            metrics["Train Samples"] = (
                result["train_samples"]
            )

            metrics["Test Samples"] = (
                result["test_samples"]
            )

            metrics["Best Model"] = (
                result["best_model"]
            )

            metrics["Is Best Model"] = (
                metrics["Model"] == result["best_model"]
            )

            metric_rows.append(
                metrics
            )

            best_row = metrics.loc[metrics["Is Best Model"]].iloc[0]
            logger.info(
                "  [%d/%d] %s / %s: melhor modelo=%s | R2=%.3f | MAE=%.3f | "
                "Within2=%.1f%% | %.1fs",
                attr_idx, len(attribute_columns), current_position, target,
                result["best_model"], best_row["R2"], best_row["MAE"],
                best_row["Within2"] * 100, attr_elapsed,
            )

            # ====================================================
            # IMPORTANCE
            # ====================================================

            for item in result[
                "feature_importance"
            ]:

                importance_rows.append(
                    {
                        "Position": current_position,
                        "Attribute": target,
                        "Attribute Category":
                            get_attribute_category(
                                target
                            ),
                        "Model":
                            result["best_model"],
                        **item,
                    }
                )

            # ====================================================
            # PREDICTIONS
            # ====================================================

            pred = (
                result["predictions"]
                .copy()
            )

            pred.insert(
                2,
                "Position",
                current_position,
            )

            pred.insert(
                3,
                "Best Model",
                result["best_model"],
            )

            prediction_frames.append(
                pred
            )

    total_elapsed = time.perf_counter() - experiment_start
    logger.info(
        "Experimento concluído em %.1fs: %d combinação(ões) treinada(s), "
        "%d pulada(s).",
        total_elapsed, combinations_done, combinations_skipped,
    )

    # ============================================================
    # RESULTADOS
    # ============================================================

    if not metric_rows:
        raise ValueError(
            "Nenhum modelo pôde ser treinado."
        )

    metrics_df = pd.concat(
        metric_rows,
        ignore_index=True,
    )

    # Melhor modelo de cada combinação aparece primeiro, o que torna
    # o CSV navegável sem precisar filtrar: quem só quer os melhores
    # resultados filtra "Is Best Model" == True.
    metrics_df = metrics_df.sort_values(
        ["Position", "Attribute", "Is Best Model"],
        ascending=[True, True, False],
    ).reset_index(drop=True)

    importance_df = pd.DataFrame(
        importance_rows
    )

    predictions_df = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    n_best = int(metrics_df["Is Best Model"].sum())
    logger.info(
        "Resumo final: %d linha(s) de métricas (%d melhores modelos), "
        "%d linha(s) de importância de features, %d previsão(ões) individuais.",
        len(metrics_df), n_best, len(importance_df), len(predictions_df),
    )

    return {
        "ATTRIBUTE_MODELS_METRICS":
            metrics_df,

        "ATTRIBUTE_FEATURE_IMPORTANCE":
            importance_df,

        "ATTRIBUTE_PREDICTIONS_ALL":
            predictions_df,
    }