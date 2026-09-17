from __future__ import annotations

import pandas as pd

from helpers.positioning import (
    CANONICAL_POSITIONS,
    position_matches,
)


def build_position_summary(
    df: pd.DataFrame,
    position_column: str = "Best Position",
) -> pd.DataFrame:

    if position_column not in df.columns:
        raise ValueError(
            f"Coluna não encontrada: "
            f"{position_column}"
        )

    rows = []

    for position in CANONICAL_POSITIONS:

        mask = df[position_column].apply(
            lambda value: position_matches(
                value,
                position,
            )
        )

        subset = df.loc[mask]

        rows.append(
            {
                "Position": position,
                "Players": len(subset),
                "Average Rating": (
                    subset["Rating"].mean()
                    if "Rating" in subset.columns
                    else float("nan")
                ),
            }
        )

    return pd.DataFrame(rows)