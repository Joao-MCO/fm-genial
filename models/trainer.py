from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import KFold, train_test_split

from analysis.attribute_features import add_attribute_features
from analysis.division_features import add_division_features
from analysis.feature_selection import (
    get_selectable_features,
    rank_features_for_target,
    select_features_for_target,
)
from config.settings import (
    DEFAULT_CORRELATION_THRESHOLD,
    DEFAULT_CV,
    DEFAULT_TEST_SIZE,
    RANDOM_STATE,
)
from models.attribute_model import build_model_candidates
from models.evaluator import (
    calculate_feature_importance,
    cross_validate_model,
    evaluate_by_attribute_range,
    evaluate_predictions,
)
from models.tuner import tune_model

logger = logging.getLogger(__name__)


def _prepare_train_test(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    *,
    features: list[str],
    target: str,
    position: str | None,
    prior_strength: float,
    division_column: str = "Division",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepara as features de treino e teste.

    Para features de divisão:
    - treino utiliza leave-one-out;
    - teste utiliza exclusivamente o conjunto de treino
      como referência.
    """

    X_train = train_data[features].copy()
    X_test = test_data[features].copy()

    if not position:
        return X_train, X_test

    train_features, test_features = add_division_features(
        train_data.copy(),
        test_data.copy(),
        reference_train=train_data.copy(),
        target=target,
        position=position,
        prior_strength=prior_strength,
        division_column=division_column,
    )

    division_features = [
        column
        for column in train_features.columns
        if column.startswith("Division ")
    ]

    for column in division_features:
        X_train[column] = train_features[column]
        X_test[column] = test_features[column]

    return X_train, X_test


def _cross_validate_with_division(
    model,
    train_data: pd.DataFrame,
    features: list[str],
    target: str,
    position: str,
    selected_features: list[str],
    *,
    cv: int,
    random_state: int,
    prior_strength: float,
    division_column: str = "Division",
    correlation_threshold: float = DEFAULT_CORRELATION_THRESHOLD,
    include_economic_features: bool = False,
) -> dict[str, float]:
    """
    Cross-validation com features derivadas da divisão.

    Em cada fold, as médias da divisão são calculadas
    exclusivamente com os jogadores do fold de treino.

    Isso evita vazamento de informação entre treino e validação.
    """

    if cv <= 1 or len(train_data) < cv:
        return {
            "CV_MAE": np.nan,
            "CV_RMSE": np.nan,
            "CV_R2": np.nan,
        }

    folds = KFold(
        n_splits=cv,
        shuffle=True,
        random_state=random_state,
    )

    maes: list[float] = []
    rmses: list[float] = []
    r2s: list[float] = []

    for fold_train_idx, fold_val_idx in folds.split(train_data):

        fold_train = train_data.iloc[
            fold_train_idx
        ].copy()

        fold_val = train_data.iloc[
            fold_val_idx
        ].copy()

        # --------------------------------------------------------
        # As features de divisão do fold são calculadas somente
        # com fold_train.
        # --------------------------------------------------------

        X_fold, X_val = _prepare_train_test(
            fold_train,
            fold_val,
            features=features,
            target=target,
            position=position,
            prior_strength=prior_strength,
            division_column=division_column,
        )

        y_fold = pd.to_numeric(
            fold_train[target],
            errors="coerce",
        )

        y_val = pd.to_numeric(
            fold_val[target],
            errors="coerce",
        )

        valid_fold = y_fold.notna()
        valid_val = y_val.notna()

        X_fold = X_fold.loc[valid_fold]
        y_fold = y_fold.loc[valid_fold]

        X_val = X_val.loc[valid_val]
        y_val = y_val.loc[valid_val]

        # A selecao e refeita dentro de cada fold para evitar leakage.
        fold_selected = select_features_for_target(
            X_fold,
            y_fold,
            correlation_threshold=correlation_threshold,
            max_features=min(30, len(X_fold.columns)),
            random_state=random_state,
            include_economic=include_economic_features,
        )
        available_features = [
            feature for feature in fold_selected
            if feature in X_fold.columns and feature in X_val.columns
        ]
        if not available_features:
            available_features = get_selectable_features(
                X_fold,
                include_economic=include_economic_features,
            )

        X_fold = X_fold[available_features]
        X_val = X_val[available_features]

        estimator = clone(model)

        estimator.fit(
            X_fold,
            y_fold,
        )

        prediction_raw = estimator.predict(
            X_val,
        )

        prediction = np.clip(
            np.rint(prediction_raw),
            1,
            20,
        ).astype(int)

        metrics = evaluate_predictions(
            y_val,
            prediction,
        )

        maes.append(metrics["MAE"])
        rmses.append(metrics["RMSE"])
        r2s.append(metrics["R2"])

    return {
        "CV_MAE": float(np.mean(maes)),
        "CV_RMSE": float(np.mean(rmses)),
        "CV_R2": float(np.mean(r2s)),
    }


def train_and_compare(
    data: pd.DataFrame,
    features: list[str],
    attribute: str,
    *,
    test_size: float = DEFAULT_TEST_SIZE,
    cv: int = DEFAULT_CV,
    alpha: float = 1.0,
    correlation_threshold: float = DEFAULT_CORRELATION_THRESHOLD,
    tune: bool = False,
    random_state: int = RANDOM_STATE,
    permutation_repeats: int = 8,
    division_column: str | None = None,
    position: str | None = None,
    division_prior: float = 20.0,
    include_economic_features: bool = False,
) -> dict:
    """
    Treina e compara os modelos para um atributo.

    Fluxo:

        dataset
            ↓
        train/test split
            ↓
        features de divisão
            ↓
        seleção de features
            ↓
        treinamento dos modelos
            ↓
        cross-validation
            ↓
        escolha pelo CV_R2
            ↓
        avaliação final no TESTE
    """

    # ============================================================
    # TARGET
    # ============================================================

    if attribute not in data.columns:
        raise ValueError(
            f"Atributo '{attribute}' não encontrado nos dados."
        )

    y = pd.to_numeric(
        data[attribute],
        errors="coerce",
    )

    valid = y.notna()

    data = data.loc[valid].copy()
    data[attribute] = y.loc[valid]

    y = data[attribute]

    if len(data) < 2 or y.nunique() < 2:
        raise ValueError(
            f"Dados insuficientes para {attribute}."
        )

    # ============================================================
    # FEATURES
    # ============================================================

    missing_features = [
        feature
        for feature in features
        if feature not in data.columns
    ]

    if missing_features:
        raise ValueError(
            f"Features não encontradas para {attribute}: "
            f"{missing_features}"
        )

    # ============================================================
    # TRAIN / TEST
    # ============================================================

    train_data, test_data = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
    )

    train_data = train_data.copy()
    test_data = test_data.copy()

    y_train = train_data[attribute]
    y_test = test_data[attribute]

    # ============================================================
    # DIVISION FEATURES
    # ============================================================

    use_division_features = bool(
        division_column
        and position
        and division_column in train_data.columns
    )

    if use_division_features:
        X_train, X_test = _prepare_train_test(
            train_data,
            test_data,
            features=features,
            target=attribute,
            position=position,
            prior_strength=division_prior,
            division_column=division_column,
        )

    else:
        X_train, X_test = _prepare_train_test(
            train_data,
            test_data,
            features=features,
            target=attribute,
            position=None,
            prior_strength=division_prior,
        )

    # ============================================================
    # FEATURES ESPECÍFICAS DO ATRIBUTO
    # ============================================================

    X_train = add_attribute_features(
        X_train,
        attribute,
    )
    X_test = add_attribute_features(
        X_test,
        attribute,
    )

    # ============================================================
    # FEATURE SELECTION
    # ============================================================

    selected_features = select_features_for_target(
        X_train,
        y_train,
        correlation_threshold=correlation_threshold,
        max_features=min(30, len(X_train.columns)),
        random_state=random_state,
        include_economic=include_economic_features,
    )

    if not selected_features:
        selected_features = get_selectable_features(
            X_train,
            include_economic=include_economic_features,
        )

    logger.info(
        "[%s] Features selecionadas: %d/%d",
        attribute,
        len(selected_features),
        len(X_train.columns),
    )

    feature_selection = rank_features_for_target(
        X_train,
        y_train,
        random_state=random_state,
        include_economic=include_economic_features,
    )

    X_train_selected = X_train[
        selected_features
    ].copy()

    X_test_selected = X_test[
        selected_features
    ].copy()

    # ============================================================
    # MODELOS
    # ============================================================

    rows: list[dict] = []
    fitted: dict = {}

    candidates = build_model_candidates(
        alpha=alpha,
        random_state=random_state,
    )

    # ============================================================
    # BASELINE
    # ============================================================

    baseline = DummyRegressor(
        strategy="mean",
    )

    baseline.fit(
        X_train_selected,
        y_train,
    )

    baseline_pred_raw = baseline.predict(
        X_test_selected,
    )

    baseline_pred = np.clip(
        np.rint(baseline_pred_raw),
        1,
        20,
    ).astype(int)

    baseline_metrics = evaluate_predictions(
        y_test,
        baseline_pred,
    )

    rows.append(
        {
            "Model": "BaselineMean",
            **baseline_metrics,
            "CV_MAE": np.nan,
            "CV_RMSE": np.nan,
            "CV_R2": 0.0,
            "Tuned": False,
            "Tuning CV R2": np.nan,
        }
    )

    # ============================================================
    # MODELOS
    # ============================================================

    for name, wrapper in candidates.items():

        model = wrapper.pipeline

        params = None
        tuning_cv_r2 = np.nan

        # --------------------------------------------------------
        # TUNING
        # --------------------------------------------------------

        if tune:
            def build_tuning_fold(
                fold_train_idx: np.ndarray,
                fold_val_idx: np.ndarray,
            ):
                fold_train = train_data.iloc[fold_train_idx].copy()
                fold_val = train_data.iloc[fold_val_idx].copy()

                X_fold, X_val = _prepare_train_test(
                    fold_train,
                    fold_val,
                    features=features,
                    target=attribute,
                    position=position if use_division_features else None,
                    prior_strength=division_prior,
                    division_column=division_column or "Division",
                )

                X_fold = add_attribute_features(
                    X_fold,
                    attribute,
                )
                X_val = add_attribute_features(
                    X_val,
                    attribute,
                )

                y_fold = pd.to_numeric(
                    fold_train[attribute],
                    errors="coerce",
                )
                y_val = pd.to_numeric(
                    fold_val[attribute],
                    errors="coerce",
                )

                valid_fold = y_fold.notna()
                valid_val = y_val.notna()

                X_fold = X_fold.loc[valid_fold]
                X_val = X_val.loc[valid_val]
                y_fold = y_fold.loc[valid_fold]
                y_val = y_val.loc[valid_val]

                fold_selected = select_features_for_target(
                    X_fold,
                    y_fold,
                    correlation_threshold=correlation_threshold,
                    max_features=min(30, len(X_fold.columns)),
                    random_state=random_state,
                    include_economic=include_economic_features,
                )
                available = [
                    feature for feature in fold_selected
                    if feature in X_fold.columns and feature in X_val.columns
                ]
                if not available:
                    available = get_selectable_features(
                        X_fold,
                        include_economic=include_economic_features,
                    )

                return (
                    X_fold[available],
                    X_val[available],
                    y_fold,
                    y_val,
                )

            model, params, tuning_cv_r2 = tune_model(
                name,
                model,
                X_train_selected,
                y_train,
                random_state,
                fold_builder=build_tuning_fold,
                cv=min(cv, 3),
            )
        else:
            model.fit(
                X_train_selected,
                y_train,
            )

        # --------------------------------------------------------
        # TEST
        # --------------------------------------------------------

        prediction_raw = model.predict(
            X_test_selected,
        )

        prediction = np.clip(
            np.rint(prediction_raw),
            1,
            20,
        ).astype(int)

        metrics = evaluate_predictions(
            y_test,
            prediction,
        )

        # --------------------------------------------------------
        # CROSS-VALIDATION
        # --------------------------------------------------------

        if use_division_features:

            cv_metrics = _cross_validate_with_division(
                model,
                train_data,
                features,
                attribute,
                position,
                selected_features,
                cv=cv,
                random_state=random_state,
                prior_strength=division_prior,
                division_column=division_column or "Division",
                correlation_threshold=correlation_threshold,
                include_economic_features=include_economic_features,
            )

        else:

            cv_metrics = cross_validate_model(
                model,
                X_train_selected,
                y_train,
                cv=cv,
                random_state=random_state,
            )

        # --------------------------------------------------------
        # MÉTRICAS POR FAIXA
        # --------------------------------------------------------

        range_metrics = evaluate_by_attribute_range(
            y_test,
            prediction,
        )

        rows.append(
            {
                "Model": name,
                **metrics,
                **cv_metrics,
                **range_metrics,
                "Tuned": bool(tune),
                "Tuning CV R2": tuning_cv_r2,
            }
        )

        logger.debug(
            "    %s: R2=%.3f | CV_R2=%.3f | MAE=%.3f | Bias=%+.3f",
            name, metrics["R2"], cv_metrics.get("CV_R2", float("nan")),
            metrics["MAE"], metrics["Bias"],
        )

        fitted[name] = {
            "model": model,
            "prediction": prediction,
            "prediction_raw": prediction_raw,
            "params": params,
        }

    # ============================================================
    # COMPARAÇÃO
    # ============================================================

    metrics_df = pd.DataFrame(rows)

    comparable = metrics_df[
        metrics_df["Model"] != "BaselineMean"
    ].copy()

    if comparable.empty:
        raise ValueError(
            f"Nenhum modelo foi treinado para {attribute}."
        )

    # ============================================================
    # MELHOR MODELO
    # ============================================================
    #
    # O TESTE não participa da escolha.
    #
    # Primeiro:
    #     maior CV_R2
    #
    # Em caso de empate:
    #     menor CV_MAE
    #
    # O TESTE fica reservado para a avaliação final.
    # ============================================================

    best_row = (
        comparable
        .sort_values(
            ["CV_R2", "CV_MAE"],
            ascending=[False, True],
        )
        .iloc[0]
    )

    best_name = best_row["Model"]

    best_result = fitted[best_name]

    best_model = best_result["model"]
    best_pred = best_result["prediction"]
    best_pred_raw = best_result["prediction_raw"]
    best_params = best_result["params"]

    # ============================================================
    # FEATURE IMPORTANCE
    # ============================================================

    importance = calculate_feature_importance(
        best_model,
        X_test_selected,
        y_test,
        selected_features,
        random_state=random_state,
        n_repeats=permutation_repeats,
    )

    # ============================================================
    # PREDICTIONS
    # ============================================================

    predictions = pd.DataFrame(
        {
            "Unique ID": (
                test_data["Unique ID"].to_numpy()
                if "Unique ID" in test_data.columns
                else np.arange(len(test_data))
            ),
            "Season": (
                test_data["Season"].to_numpy()
                if "Season" in test_data.columns
                else test_data["Temporada"].to_numpy()
                if "Temporada" in test_data.columns
                else ""
            ),
            "Player": (
                test_data["Player"].to_numpy()
                if "Player" in test_data.columns
                else test_data["Player_stats"].to_numpy()
                if "Player_stats" in test_data.columns
                else test_data["Player_attributes"].to_numpy()
                if "Player_attributes" in test_data.columns
                else ""
            ),
            "Attribute": attribute,
            "Actual": y_test.to_numpy(),
            "Predicted": best_pred,
            "Predicted Raw": best_pred_raw,
        }
    )

    if "Season" in predictions.columns and predictions["Season"].astype(str).str.strip().ne("").any():
        predictions["Sample ID"] = (
            predictions["Unique ID"].astype(str)
            + ":"
            + predictions["Season"].astype(str)
        )

    predictions["Absolute Error"] = (
        predictions["Actual"]
        - predictions["Predicted"]
    ).abs()

    # ============================================================
    # RESULTADO
    # ============================================================

    return {
        "metrics": metrics_df,
        "best_model": best_name,
        "best_estimator": best_model,
        "best_params": best_params,
        "selected_features": selected_features,
        "feature_selection": feature_selection,
        "feature_importance": importance,
        "predictions": predictions,
        "train_samples": len(train_data),
        "test_samples": len(test_data),
    }