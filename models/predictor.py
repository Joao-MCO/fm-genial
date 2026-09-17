from __future__ import annotations

import numpy as np
import pandas as pd


def build_predictions(
    df: pd.DataFrame,
    y_true,
    y_pred,
    *,
    attribute: str,
    position: str,
) -> pd.DataFrame:

    predicted = np.clip(
        np.rint(y_pred),
        1,
        20,
    ).astype(int)

    result = pd.DataFrame(
        {
            "Unique ID": df[
                "Unique ID"
            ].values,

            "Player": df[
                "Player"
            ].values,

            "Position": position,

            "Attribute": attribute,

            "Actual": y_true,

            "Predicted": predicted,
        }
    )

    result["Absolute Error"] = (
        result["Actual"]
        - result["Predicted"]
    ).abs()

    return result