from __future__ import annotations

import numpy as np
import pandas as pd


# Features de desempenho mais diretamente relacionadas a cada atributo.
# Elas continuam convivendo com todas as features gerais.
ATTRIBUTE_FEATURE_GROUPS = {
    "Finishing": (
        "Goals/90",
        "xG/90",
        "NP-xG/90",
        "xG/shot",
        "Shots/90",
        "Shots on Target/90",
        "ShotsOnTargetRate",
        "Finishing_ShotConversion",
        "Finishing_OnTargetConversion",
    ),

    "Passing": (
        "Passes Attempted/90",
        "Passes Completed/90",
        "PassCompletionRateDerived",
        "Key Passes/90",
        "Open Play Key Passes per 90 minutes/90",
        "Chances Created per 90",
        "xA/90",
        "Assists/90",
        "Passing_KeyPassShare",
    ),

    "Crossing": (
        "Crosses Attempted/90",
        "Crosses Completed/90",
        "CrossCompletionRate",
        "Open Play Crosses Attempted/90",
        "Open Play Crosses Completed/90",
        "OpenPlayCrossCompletionRate",
    ),

    "Dribbling": (
        "Dribbles/90",
        "Possession Lost per 90",
        "Possession Won per 90",
        "Dribbling_WinToLoss",
    ),

    "Heading": (
        "Headers Attempted/90",
        "Headers Won/90",
        "HeaderWinRate",
        "Key Headers per 90",
        "height_cm",
    ),

    "Tackling": (
        "Tackles Attempted/90",
        "Tackled Completed/90",
        "TackleSuccessRate",
        "Key Tackles/90",
        "Interceptions/90",
        "Tackling_InterceptionRate",
    ),

    "Marking": (
        "Interceptions/90",
        "Clearances/90",
        "Tackles Attempted/90",
        "TackleSuccessRate",
        "Blk/90",
        "Marking_DefensiveActions",
    ),

    "Long Shots": (
        "Shots From Outside The Box Per 90 minutes",
        "Shots/90",
        "xG/shot",
        "Goals From Outside The Box/90",
    ),

    "Corners": (
        "Corners/90",
        "Crosses Attempted/90",
        "Key Passes/90",
    ),

    "Free Kick Taking": (
        "Free Kick Shots/90",
        "xG/shot",
        "Shots From Outside The Box Per 90 minutes",
    ),

    "Penalty Taking": (
        "Penalties Taken/90",
        "Penalties Scored/90",
        "Penalty_SuccessRate",
    ),

    "Composure": (
        "Mistakes Leading to Goals/90",
        "PassCompletionRateDerived",
        "ShotsOnTargetRate",
    ),

    "Off The Ball": (
        "Goals/90",
        "Shots/90",
        "Clear Cut Chances Created/90",
        "Off/90",
    ),

    "Positioning": (
        "Interceptions/90",
        "Clearances/90",
        "Key Tackles/90",
    ),
}

def _ratio(
    df: pd.DataFrame,
    numerator: str,
    denominator: str,
) -> pd.Series | None:
    if numerator not in df.columns or denominator not in df.columns:
        return None

    num = pd.to_numeric(df[numerator], errors="coerce")
    den = pd.to_numeric(df[denominator], errors="coerce").replace(0, np.nan)
    return num.div(den)


def add_attribute_features(
    X: pd.DataFrame,
    target: str,
) -> pd.DataFrame:
    """Add leakage-safe features derived only from match/performance stats."""
    X = X.copy()

    derived: dict[str, pd.Series | None] = {
        "Finishing_ShotConversion": _ratio(X, "Goals", "Shots"),
        "Finishing_OnTargetConversion": _ratio(X, "Goals", "Shots on Target"),
        "Passing_KeyPassShare": _ratio(X, "Key Passes", "Passes Attempted"),
        "Dribbling_WinToLoss": _ratio(X, "Possession Won per 90", "Possession Lost per 90"),
        "Tackling_InterceptionRate": _ratio(X, "Interceptions", "Tackles Attempted"),
        "Marking_DefensiveActions": None,
        "Penalty_SuccessRate": _ratio(X, "Penalties Scored", "Penalties Taken"),
    }

    if "Marking" == target:
        parts = [
            X[column]
            for column in (
                "Interceptions/90",
                "Clearances/90",
                "Tackles Attempted/90",
                "Blk/90",
            )
            if column in X.columns
        ]
        if parts:
            derived["Marking_DefensiveActions"] = sum(parts)

    allowed = set(ATTRIBUTE_FEATURE_GROUPS.get(target, ()))

    for name, values in derived.items():
        if values is not None and name in allowed:
            X[name] = pd.to_numeric(values, errors="coerce")

    return X
