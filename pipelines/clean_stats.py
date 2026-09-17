from __future__ import annotations

import pandas as pd

from analysis.feature_engineering import build_features


def build_clean_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gera uma versão analítica do dataset de stats.

    Mantém todas as colunas originais do dataset já limpo e
    adiciona somente as métricas derivadas de futebol.

    Não altera o DataFrame original.
    """
    clean_stats = df.copy()

    features = build_features(df)

    # Adiciona somente features que ainda não existem no dataset original.
    derived_features = features.drop(
        columns=[
            column
            for column in features.columns
            if column in clean_stats.columns
        ],
        errors="ignore",
    )

    if not derived_features.empty:
        clean_stats = pd.concat(
            [
                clean_stats,
                derived_features,
            ],
            axis=1,
        )

    return clean_stats


def run_clean_stats_pipeline(
    stats: pd.DataFrame,
) -> pd.DataFrame:
    """
    Executa o pipeline de geração do CLEAN_STATS.
    """
    return build_clean_stats(stats)