from __future__ import annotations

import argparse
import logging
import os
import time

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

from config.settings import (
    DEFAULT_ALPHA, DEFAULT_CORRELATION_THRESHOLD, DEFAULT_CV, DEFAULT_DIVISION_PRIOR,
    DEFAULT_MIN_SAMPLES, DEFAULT_PERMUTATION_REPEATS, DEFAULT_TEST_SIZE,
)
from helpers.exporter import export_csv
from helpers.loader import load_and_clean
from helpers.resolvers import resolve_attributes_path, resolve_csv_path
from paths import ATTRIBUTES_DIR, STATS_DIR
from pipelines.attributes import run_attribute_experiment

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Experimentos de ML com dados do Football Manager.")
    parser.add_argument("--file", dest="file_flag", help="CSV de stats individual. Ignorado se --seasons for usado.")
    parser.add_argument(
        "--seasons",
        default=None,
        help=(
            "Temporadas a carregar, separadas por vírgula (ex.: SEASON_01,SEASON_02). "
            "Sem --file/--seasons, detecta automaticamente databases/stats/SEASON_*.csv."
        ),
    )
    parser.add_argument("--attributes-file", default=None, help="CSV de attributes; por padrão, usa attributes/<mesmo nome>.csv")
    parser.add_argument("--mode", choices=["rating", "attributes"], default="attributes")
    parser.add_argument("--position", default=None)
    parser.add_argument("--attribute", "--target", dest="attribute", default=None, help="Atributo específico; sem valor, executa todos.")
    parser.add_argument("--cv", type=int, default=DEFAULT_CV)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES)
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--correlation-threshold", type=float, default=DEFAULT_CORRELATION_THRESHOLD)
    parser.add_argument("--permutation-repeats", type=int, default=DEFAULT_PERMUTATION_REPEATS)
    parser.add_argument("--tune", action="store_true", help="Executa RandomizedSearchCV (mais lento).")
    parser.add_argument("--division-prior", type=float, default=DEFAULT_DIVISION_PRIOR, help="Força do shrinkage das médias da divisão.")
    parser.add_argument("--save-clean-stats", action="store_true", help="Também salva STATS_CLEAN.csv (dados após limpeza, sem resultado de modelo).")
    parser.add_argument("-q", "--quiet", action="store_true", help="Mostra só avisos e erros (silencia o progresso).")
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


def resolve_stats_file(filename: str | None) -> str:
    if filename:
        return resolve_csv_path(filename)
    default = os.path.join(STATS_DIR, "ALL.csv")
    if os.path.isfile(default):
        return default
    raise FileNotFoundError("Informe o arquivo de stats.")


def resolve_season_paths(seasons: str | None) -> list[tuple[str, str, str]]:
    """Resolve pares stats/attributes por temporada.

    Cada temporada continua sendo uma amostra independente no dataset final.
    O identificador da temporada é preservado na coluna ``Season``.
    """
    if seasons:
        names = [item.strip() for item in seasons.split(",") if item.strip()]
    else:
        pattern = os.path.join(STATS_DIR, "SEASON_*.csv")
        names = [os.path.splitext(os.path.basename(path))[0] for path in sorted(__import__("glob").glob(pattern))]

    if not names:
        raise FileNotFoundError(
            "Nenhuma temporada encontrada. Use --seasons SEASON_01,SEASON_02,... "
            "ou informe --file para uma base única."
        )

    resolved: list[tuple[str, str, str]] = []
    for season in names:
        filename = season if season.lower().endswith(".csv") else f"{season}.csv"
        stats_path = resolve_csv_path(os.path.join("stats", filename))
        attrs_path = resolve_attributes_path(stats_path, None)
        season_name = os.path.splitext(os.path.basename(stats_path))[0]
        resolved.append((season_name, stats_path, attrs_path))

    return resolved


def load_seasons(season_paths: list[tuple[str, str, str]], *, require_rating: bool = False):
    stats_frames = []
    attrs_frames = []

    for season, stats_path, attrs_path in season_paths:
        logger.info("Carregando %s: stats=%s | attributes=%s", season, stats_path, attrs_path)
        stats = load_and_clean(stats_path, require_rating=require_rating)
        attrs = pd.read_csv(attrs_path, sep=";", encoding="utf-8-sig")

        stats["Season"] = season
        attrs["Season"] = season

        stats_frames.append(stats)
        attrs_frames.append(attrs)
        logger.info("  %s: %d stats / %d attributes", season, len(stats), len(attrs))

    return (
        pd.concat(stats_frames, ignore_index=True),
        pd.concat(attrs_frames, ignore_index=True),
    )


def run_rating_experiment(df: pd.DataFrame):
    if "Rating" not in df.columns:
        raise ValueError("Coluna Rating não encontrada.")
    numeric = df.select_dtypes(include="number").drop(columns=["Rating"], errors="ignore")
    y = pd.to_numeric(df["Rating"], errors="coerce")
    mask = y.notna()
    X = numeric.loc[mask]
    y = y.loc[mask]
    logger.info("Treinando modelo de Rating com %d amostras...", len(X))
    start = time.perf_counter()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=DEFAULT_TEST_SIZE, random_state=42)
    model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=DEFAULT_ALPHA))])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    baseline = DummyRegressor(strategy="mean").fit(X_train, y_train).predict(X_test)
    result = pd.DataFrame({"REAL": y_test.to_numpy(), "IA": pred})
    export_csv(result, "PREDICOES_RATING")
    elapsed = time.perf_counter() - start
    logger.info(
        "Rating treinado em %.1fs | Baseline R²=%.4f | R²=%.4f | MAE=%.4f | RMSE=%.4f",
        elapsed, r2_score(y_test, baseline), r2_score(y_test, pred),
        mean_absolute_error(y_test, pred), mean_squared_error(y_test, pred) ** 0.5,
    )


def main():
    args = parse_args()
    configure_logging(quiet=args.quiet, verbose=args.verbose)

    if args.cv < 2:
        raise ValueError("--cv precisa ser >= 2")
    if args.min_samples < 2:
        raise ValueError("--min-samples precisa ser >= 2")
    if not 0 < args.test_size < 1:
        raise ValueError("--test-size precisa estar entre 0 e 1")
    if args.alpha < 0:
        raise ValueError("--alpha precisa ser >= 0")
    if not 0 < args.correlation_threshold <= 1:
        raise ValueError("--correlation-threshold precisa estar entre 0 e 1")
    if args.division_prior < 0:
        raise ValueError(
            "--division-prior precisa ser >= 0"
        )

    filename = args.file_flag

    if args.seasons and filename:
        raise ValueError("Use --file ou --seasons, não os dois.")

    if args.seasons or not filename:
        season_paths = resolve_season_paths(args.seasons)
        stats, attributes = load_seasons(
            season_paths,
            require_rating=args.mode == "rating",
        )
        logger.info(
            "Dataset multi-temporada carregado: %d temporada(s), %d jogador-temporada(s).",
            len(season_paths), len(stats),
        )
    else:
        stats_path = resolve_stats_file(filename)
        logger.info("Carregando e limpando dados de %s...", stats_path)
        stats = load_and_clean(stats_path, require_rating=args.mode == "rating")
        logger.info("Dados carregados: %d jogador(es).", len(stats))
        attributes_path = resolve_attributes_path(stats_path, args.attributes_file)
        attributes = pd.read_csv(attributes_path, sep=";", encoding="utf-8-sig")

    if args.save_clean_stats:
        export_csv(stats, "STATS_CLEAN")

    if args.mode == "rating":
        run_rating_experiment(stats)
        return

    run_start = time.perf_counter()
    outputs = run_attribute_experiment(
        stats,
        attributes,
        position=args.position,
        attribute=args.attribute,
        cv=args.cv,
        alpha=args.alpha,
        min_samples=args.min_samples,
        test_size=args.test_size,
        correlation_threshold=args.correlation_threshold,
        tune=args.tune,
        permutation_repeats=args.permutation_repeats,
        division_prior=args.division_prior,
    )
    run_elapsed = time.perf_counter() - run_start

    suffix = args.position.upper() if args.position else "ALL_POSITIONS"
    for name, frame in outputs.items():
        export_csv(frame, f"{name}_{suffix}")

    metrics = outputs["ATTRIBUTE_MODELS_METRICS"]
    best = metrics.loc[metrics["Is Best Model"]]

    logger.info("Tempo total do experimento: %.1fs", run_elapsed)
    print("\n=== MELHOR MODELO POR ATRIBUTO ===")
    print(
        best[
            [
                "Position", "Attribute", "Model", "R2", "MAE", "MedianAE",
                "Bias", "RMSE", "Within1", "Within2", "Within3",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()