from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402


from helpers.exporter import export_csv  # noqa: E402
from helpers.loader import load_and_clean  # noqa: E402
from main import resolve_season_paths, resolve_stats_file  # noqa: E402
from pipelines.clean_stats import build_clean_stats  # noqa: E402

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Gera STATS_CLEAN.csv: os stats já limpos (parsing de dinheiro, "
            "altura, posições etc.) mais as métricas derivadas de futebol "
            "(/90, taxas de conversão etc.), sem treinar nenhum modelo."
        )
    )
    parser.add_argument("--file", dest="file_flag", help="CSV de stats individual. Ignorado se --seasons for usado.")
    parser.add_argument(
        "--seasons",
        default=None,
        help=(
            "Temporadas a carregar, separadas por vírgula (ex.: SEASON_01,SEASON_02). "
            "Sem --file/--seasons, detecta automaticamente databases/stats/SEASON_*.csv."
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


def load_stats_only(season_paths: list[tuple[str, str, str]]) -> pd.DataFrame:
    """Carrega e concatena só os stats de cada temporada (sem attributes)."""
    frames = []
    for season, stats_path, _attrs_path in season_paths:
        logger.info("Carregando %s: stats=%s", season, stats_path)
        stats = load_and_clean(stats_path, require_rating=False)
        stats["Season"] = season
        frames.append(stats)
        logger.info("  %s: %d stats", season, len(stats))
    return pd.concat(frames, ignore_index=True)


def main():
    args = parse_args()
    configure_logging(quiet=args.quiet, verbose=args.verbose)

    filename = args.file_flag

    if args.seasons and filename:
        raise ValueError("Use --file ou --seasons, não os dois.")

    if args.seasons or not filename:
        season_paths = resolve_season_paths(args.seasons)
        stats = load_stats_only(season_paths)
        logger.info(
            "Dataset multi-temporada carregado: %d temporada(s), %d jogador-temporada(s).",
            len(season_paths), len(stats),
        )
    else:
        stats_path = resolve_stats_file(filename)
        logger.info("Carregando e limpando dados de %s...", stats_path)
        stats = load_and_clean(stats_path, require_rating=False)
        logger.info("Dados carregados: %d jogador(es).", len(stats))

    logger.info("Gerando métricas derivadas...")
    clean = build_clean_stats(stats)

    path = export_csv(clean, "STATS_CLEAN")

    print("\n=== STATS_CLEAN GERADO ===")
    print(f"Linhas: {len(clean)}")
    print(f"Colunas: {clean.shape[1]} (originais: {stats.shape[1]})")
    print(f"Arquivo: {path}")


if __name__ == "__main__":
    main()