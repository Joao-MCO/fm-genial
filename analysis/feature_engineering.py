from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.feature_selection import EXCLUDED_FEATURES


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
    multiplier: float = 1.0,
) -> pd.Series:
    denominator = _numeric(denominator).replace(0, np.nan)
    return _numeric(numerator).div(denominator).mul(multiplier)


def _has_per90_name(column: str) -> bool:
    name = column.lower()
    return "/90" in name or "per 90" in name or "per90" in name


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build numeric performance features and derived football metrics.

    Only numeric football-performance fields are copied/generated here.
    IDs, labels, economic metadata and other excluded fields are omitted.
    """
    values: dict[str, pd.Series] = {}

    numeric_columns = [
        c for c in df.columns
        if c not in EXCLUDED_FEATURES
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    for column in numeric_columns:
        values[column] = _numeric(df[column])

    minutes = _numeric(
        df.get("Minutes", pd.Series(index=df.index, dtype=float))
    )
    starts = _numeric(
        df.get("Starts", pd.Series(index=df.index, dtype=float))
    )

    # Generate /90 only for raw count/volume metrics.
    # Fields already named "/90", "Per 90" or "per90" are already normalized.
    rate_candidates = [
        c for c in numeric_columns
        if c not in {"Minutes", "Starts", "Age", "Height", "height_cm"}
        and not _has_per90_name(c)
        and "%" not in c
        and "Percentage" not in c
        and "Ratio" not in c
    ]

    for column in rate_candidates:
        values[f"{column}/90"] = _safe_divide(
            df[column],
            minutes,
            90,
        )

    # Existing ratios/rates.
    ratio_pairs = {
        "ShotsOnTargetRate": ("Shots on Target", "Shots"),
        "CrossCompletionRate": ("Crosses Completed", "Crosses Attempted"),
        "OpenPlayCrossCompletionRate": (
            "Open Play Crosses Completed",
            "Open Play Crosses Attempted",
        ),
        "TackleSuccessRate": ("Tackled Completed", "Tackles Attempted"),
        "HeaderWinRate": ("Headers Won", "Headers Attempted"),
        "PassCompletionRateDerived": ("Passes Completed", "Passes Attempted"),
        "KeyPassRate": ("Key Passes", "Passes Attempted"),
        "AssistRate": ("Assists", "Minutes"),
    }

    for name, (num, den) in ratio_pairs.items():
        if num in df.columns and den in df.columns:
            values[name] = _safe_divide(
                df[num],
                df[den],
                90 if den == "Minutes" else 1,
            )

    if "Distance" in df.columns:
        values["Distance/90"] = _safe_divide(df["Distance"], minutes, 90)

    if "Starts" in df.columns:
        values["Starts/90"] = _safe_divide(starts, minutes, 90)

    # ------------------------------------------------------------------
    # Additional derived football metrics.
    # ------------------------------------------------------------------
    if {"Goals", "Shots"}.issubset(df.columns):
        values["GoalConversion"] = _safe_divide(df["Goals"], df["Shots"])

    if {"Shots on Target", "Shots"}.issubset(df.columns):
        values["ShotOnTargetShare"] = _safe_divide(
            df["Shots on Target"], df["Shots"]
        )

    if {"Goals", "Shots on Target"}.issubset(df.columns):
        values["GoalPerShotOnTarget"] = _safe_divide(
            df["Goals"], df["Shots on Target"]
        )

    if {"xG", "Shots"}.issubset(df.columns):
        values["XGPerShotDerived"] = _safe_divide(df["xG"], df["Shots"])

    if {"Goals", "xG"}.issubset(df.columns):
        values["GoalsMinusXG"] = _numeric(df["Goals"]) - _numeric(df["xG"])
        values["GoalsMinusXGPer90"] = _safe_divide(
            values["GoalsMinusXG"], minutes, 90
        )

    if {"Goals", "Penalties Scored"}.issubset(df.columns):
        values["NonPenaltyGoals"] = (
            _numeric(df["Goals"]) - _numeric(df["Penalties Scored"])
        )
        values["NonPenaltyGoalsPer90"] = _safe_divide(
            values["NonPenaltyGoals"], minutes, 90
        )

    if {"Penalties Scored", "Penalties Taken"}.issubset(df.columns):
        values["PenaltyConversion"] = _safe_divide(
            df["Penalties Scored"], df["Penalties Taken"]
        )

    if "NonPenaltyGoals" in values and "Shots" in df.columns:
        values["NonPenaltyGoalConversion"] = _safe_divide(
            values["NonPenaltyGoals"], df["Shots"]
        )

    if {"Assists", "xA"}.issubset(df.columns):
        values["AssistsMinusXA"] = (
            _numeric(df["Assists"]) - _numeric(df["xA"])
        )
        values["AssistsMinusXAPer90"] = _safe_divide(
            values["AssistsMinusXA"], minutes, 90
        )

    if {"Key Passes", "Chances Created per 90"}.issubset(df.columns):
        values["KeyPassToChanceRate"] = _safe_divide(
            df["Key Passes"], df["Chances Created per 90"]
        )

    if {"Clear Cut Chances Created", "Chances Created per 90"}.issubset(df.columns):
        values["ClearCutChanceRate"] = _safe_divide(
            df["Clear Cut Chances Created"],
            df["Chances Created per 90"],
        )

    if {"Open Play Key Passes per 90", "Key Passes"}.issubset(df.columns):
        values["OpenPlayKeyPassShare"] = _safe_divide(
            df["Open Play Key Passes per 90"], df["Key Passes"]
        )

    if {
        "Open Play Crosses Attempted",
        "Crosses Attempted",
    }.issubset(df.columns):
        values["OpenPlayCrossShare"] = _safe_divide(
            df["Open Play Crosses Attempted"],
            df["Crosses Attempted"],
        )

    if {
        "Shots From Outside The Box Per 90 minutes",
        "Shots",
    }.issubset(df.columns):
        values["OutsideBoxShotRate"] = _safe_divide(
            df["Shots From Outside The Box Per 90 minutes"],
            df["Shots"],
        )

    if {
        "Goals From Outside The Box",
        "Shots From Outside The Box Per 90 minutes",
    }.issubset(df.columns):
        values["OutsideBoxGoalConversion"] = _safe_divide(
            df["Goals From Outside The Box"],
            df["Shots From Outside The Box Per 90 minutes"],
        )

    if {"Pres C", "Pres A"}.issubset(df.columns):
        values["PressureSuccessRate"] = _safe_divide(
            df["Pres C"], df["Pres A"]
        )

    defensive_components = [
        c for c in ["Tackled Completed", "Interceptions", "Blk", "Clearances"]
        if c in df.columns
    ]
    if defensive_components:
        defensive_total = sum(
            (_numeric(df[c]) for c in defensive_components),
            pd.Series(0.0, index=df.index),
        )
        values["DefensiveActionsPer90"] = _safe_divide(
            defensive_total, minutes, 90
        )

    if {"Tackled Completed", "Interceptions"}.issubset(df.columns):
        values["TackleInterceptionRatio"] = _safe_divide(
            df["Tackled Completed"], df["Interceptions"]
        )

    if "Tackled Completed" in df.columns and "DefensiveActionsPer90" in values:
        values["TackleShareDefensiveActions"] = _safe_divide(
            df["Tackled Completed"], defensive_total
        )

    if "Interceptions" in df.columns and "DefensiveActionsPer90" in values:
        values["InterceptionShareDefensiveActions"] = _safe_divide(
            df["Interceptions"], defensive_total
        )

    if {"Possession Won per 90", "Possession Lost per 90"}.issubset(df.columns):
        values["PossessionBalancePer90"] = (
            _numeric(df["Possession Won per 90"])
            - _numeric(df["Possession Lost per 90"])
        )
        values["PossessionWonLostRatio"] = _safe_divide(
            df["Possession Won per 90"],
            df["Possession Lost per 90"],
        )

    if {"Fouls Against", "Fouls Made"}.issubset(df.columns):
        values["FoulBalancePer90"] = _safe_divide(
            _numeric(df["Fouls Against"]) - _numeric(df["Fouls Made"]),
            minutes,
            90,
        )
        values["FoulsDrawnCommittedRatio"] = _safe_divide(
            df["Fouls Against"], df["Fouls Made"]
        )

    card_columns = [
        c for c in ["Yellow Cards", "Red cards"]
        if c in df.columns
    ]
    if card_columns:
        cards = sum(
            (_numeric(df[c]) for c in card_columns),
            pd.Series(0.0, index=df.index),
        )
        values["CardsPer90"] = _safe_divide(cards, minutes, 90)

    return pd.DataFrame(values, index=df.index).dropna(axis=1, how="all")
