from __future__ import annotations

import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate


def evaluate_predictions(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    error = y_pred - y_true
    absolute_error = np.abs(error)

    # Correlação de Pearson entre real e previsto.
    # Complementa o R2: mostra o quanto o modelo acompanha a
    # tendência mesmo quando a escala das previsões está "encolhida".
    if len(y_true) >= 2 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        correlation = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        correlation = float("nan")

    # MAPE: erro percentual médio. Como o atributo nunca é 0,
    # a divisão é segura.
    mape = float(np.mean(absolute_error / np.abs(y_true))) * 100

    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MedianAE": float(np.median(absolute_error)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MaxError": float(np.max(absolute_error)),
        "MAPE": mape,
        "R2": float(r2_score(y_true, y_pred)),
        "Correlation": correlation,

        # Bias: erro médio COM sinal. Positivo = modelo superestima
        # em média; negativo = subestima. Diferente do MAE, que não
        # tem sinal e não revela viés sistemático.
        "Bias": float(np.mean(error)),
        "ErrorStd": float(np.std(error)),

        # Acerto exato e dentro de N pontos de erro.
        "ExactAccuracy": float(np.mean(absolute_error == 0)),
        "Within1": float(np.mean(absolute_error <= 1)),
        "Within2": float(np.mean(absolute_error <= 2)),
        "Within3": float(np.mean(absolute_error <= 3)),

        # Erro grosseiro: previsões que erram por mais de 5 pontos
        # numa escala de 20 já são inúteis na prática.
        "GrossError_Over5": float(np.mean(absolute_error > 5)),
    }
    
def evaluate_by_attribute_range(
    y_true,
    y_pred,
) -> dict[str, float]:

    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    if len(y_true) == 0:
        return {}

    q1, q2 = np.nanquantile(
        y_true,
        [1 / 3, 2 / 3],
    )

    ranges = {
        "Low": y_true <= q1,
        "Mid": (
            (y_true > q1)
            & (y_true <= q2)
        ),
        "High": y_true > q2,
    }

    result = {}

    for name, mask in ranges.items():

        if not mask.any():
            result[f"{name}_MAE"] = float("nan")
            result[f"{name}_R2"] = float("nan")
            result[f"{name}_Bias"] = float("nan")
            result[f"{name}_Within1"] = float("nan")
            continue

        metrics = evaluate_predictions(
            y_true[mask],
            y_pred[mask],
        )

        result[f"{name}_MAE"] = metrics["MAE"]
        result[f"{name}_R2"] = metrics["R2"]

        # Bias por faixa é o que mais evidencia "regressão à média":
        # espera-se Low_Bias > 0 (modelo superestima os fracos) e
        # High_Bias < 0 (modelo subestima os fortes).
        result[f"{name}_Bias"] = metrics["Bias"]
        result[f"{name}_Within1"] = metrics["Within1"]

    return result


def cross_validate_model(model, X, y, cv=5, random_state=42) -> dict[str, float]:
    if cv <= 1 or len(y) < cv:
        return {"CV_MAE": np.nan, "CV_RMSE": np.nan, "CV_R2": np.nan}
    folds = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    scores = cross_validate(
        model, X, y, cv=folds,
        scoring={"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"},
        n_jobs=1,
    )
    return {
        "CV_MAE": float(-scores["test_mae"].mean()),
        "CV_RMSE": float(-scores["test_rmse"].mean()),
        "CV_R2": float(scores["test_r2"].mean()),
    }


def calculate_feature_importance(model, X_test, y_test, feature_names, *, random_state=42, n_repeats=8):
    result = permutation_importance(
        model, X_test, y_test, scoring="r2",
        n_repeats=n_repeats, random_state=random_state, n_jobs=1,
    )
    rows = []
    for rank, (feature, mean, std) in enumerate(
        sorted(zip(feature_names, result.importances_mean, result.importances_std), key=lambda x: x[1], reverse=True), 1
    ):
        rows.append({"Feature": feature, "Importance": float(mean), "Importance Std": float(std), "Rank": rank})
    return rows