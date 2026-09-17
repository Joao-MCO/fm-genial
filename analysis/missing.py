from __future__ import annotations

import pandas as pd


def build_missing_report(
    df: pd.DataFrame,
) -> pd.DataFrame:

    total = len(df)

    rows = []

    for column in df.columns:

        missing = int(
            df[column].isna().sum()
        )

        percentage = (
            missing / total * 100
            if total
            else 0.0
        )

        rows.append(
            {
                "Column": column,
                "Missing": missing,
                "Available": total - missing,
                "Missing Percentage": percentage,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "Missing Percentage",
            ascending=False,
        )
        .reset_index(drop=True)
    )