from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipelines.merge_exports import DEFAULT_EXPORTS_DIR, merge_exports  # noqa: E402

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Une os arquivos ATT_*.csv e STATS_*.csv exportados pelo "
            "FM26PlayerExport (by vinteset) em databases/attributes/ALL.csv "
            "e databases/stats/ALL.csv."
        )
    )
    parser.add_argument(
        "--exports-dir",
        default=None,
        help=(
            "Pasta onde estão os CSVs exportados pelo FM. "
            f"Padrão: {DEFAULT_EXPORTS_DIR}"
        ),
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Mostra só avisos e erros.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Mostra logs de debug adicionais.")
    return parser.parse_args()


def configure_logging(*, quiet: bool, verbose: bool) -> None:
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    args = parse_args()
    configure_logging(quiet=args.quiet, verbose=args.verbose)

    exports_dir = args.exports_dir or DEFAULT_EXPORTS_DIR

    logger.info("Lendo exports em: %s", exports_dir)

    try:
        result = merge_exports(exports_dir)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc

    print("\n=== MERGE CONCLUÍDO ===")
    print(f"Atributos: {len(result['attributes'])} jogador(es) únicos")
    print(f"Estatísticas: {len(result['stats'])} jogador(es) únicos")
    print(
        "\nArquivos gerados em databases/attributes/ALL.csv e "
        "databases/stats/ALL.csv — já detectados automaticamente pelo "
        "main.py (sem precisar de --file/--seasons)."
    )


if __name__ == "__main__":
    main()