from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from paths import ATTRIBUTES_DIR, STATS_DIR

logger = logging.getLogger(__name__)

# Diretório padrão gerado pelo FM26PlayerExport by vinteset.
DEFAULT_EXPORTS_DIR = Path(
    r"C:\Users\Cliente\Documents\Sports Interactive\Football Manager 26\FM26PlayerExport by vinteset\Exports CSV"
)


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=";", encoding="utf-8-sig")


def _merge_group(
    files: list[Path],
    output_path: Path,
    *,
    prefix: str,
) -> pd.DataFrame:
    if not files:
        raise FileNotFoundError(f"Nenhum CSV encontrado para {prefix}.")

    frames: list[pd.DataFrame] = []
    for path in files:
        logger.info("Lendo %s...", path.name)
        frame = _read_csv(path)
        if "Unique ID" not in frame.columns:
            raise ValueError(f"O arquivo {path.name} não possui a coluna 'Unique ID'.")
        frames.append(frame)

    # A ordem dos arquivos é determinística (alfabética). Ao concatenar
    # nessa ordem e usar keep='first', a primeira ocorrência sempre vence.
    merged = pd.concat(frames, ignore_index=True, sort=False)
    before = len(merged)
    merged["Unique ID"] = merged["Unique ID"].astype(str).str.strip()
    merged = merged.drop_duplicates(subset=["Unique ID"], keep="first").reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False, sep=";", encoding="utf-8-sig")

    logger.info(
        "%s: %d arquivo(s), %d linha(s) -> %d linha(s) após deduplicação.",
        prefix,
        len(files),
        before,
        len(merged),
    )
    logger.info("Arquivo salvo: %s", output_path)
    return merged


def merge_exports(exports_dir: str | Path = DEFAULT_EXPORTS_DIR) -> dict[str, pd.DataFrame]:
    """Une todos os ATT_*.csv e STATS_*.csv do exportador do FM26.

    Os arquivos são processados em ordem alfabética e, para cada Unique ID,
    somente a primeira ocorrência é mantida.
    """
    source = Path(exports_dir).expanduser()
    if not source.is_dir():
        raise FileNotFoundError(f"Diretório de exports não encontrado: {source}")

    attribute_files = sorted(source.glob("ATT_*.csv"), key=lambda p: p.name.casefold())
    stats_files = sorted(source.glob("STATS_*.csv"), key=lambda p: p.name.casefold())

    attributes = _merge_group(
        attribute_files,
        Path(ATTRIBUTES_DIR) / "ALL.csv",
        prefix="ATRIBUTOS",
    )
    stats = _merge_group(
        stats_files,
        Path(STATS_DIR) / "ALL.csv",
        prefix="ESTATÍSTICAS",
    )

    return {
        "attributes": attributes,
        "stats": stats,
    }
