from __future__ import annotations

import logging
import os
import pandas as pd
from paths import OUTPUTS_DIR

logger = logging.getLogger(__name__)


def export_csv(df: pd.DataFrame, name: str) -> str:
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    path = os.path.join(OUTPUTS_DIR, f"{name}.csv")
    df.to_csv(path, index=False, sep=";", decimal=",", encoding="utf-8-sig")
    logger.info("Arquivo salvo: %s (%d linha(s), %d coluna(s))", path, len(df), df.shape[1])
    return path
