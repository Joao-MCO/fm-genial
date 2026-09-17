from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold, ParameterSampler, RandomizedSearchCV
from sklearn.metrics import r2_score


GRIDS = {
    "Ridge": {"model__alpha": [0.01, 0.1, 0.3, 1, 3, 10, 30, 100]},
    "RandomForest": {
        "model__n_estimators": [100, 150],
        "model__max_depth": [10, 14, 18, None],
        "model__min_samples_leaf": [2, 4, 8],
        "model__max_features": [0.5, 0.7, 1.0],
    },
    "ExtraTrees": {
        "model__n_estimators": [100, 150],
        "model__max_depth": [10, 16, 22, None],
        "model__min_samples_leaf": [2, 3, 6],
        "model__max_features": [0.5, 0.8, 1.0],
    },
    "HistGradientBoosting": {
        "model__max_iter": [200, 300, 450],
        "model__learning_rate": [0.03, 0.05, 0.08],
        "model__max_leaf_nodes": [15, 31, 63],
        "model__min_samples_leaf": [10, 20, 40],
        "model__l2_regularization": [0.1, 1.0, 5.0],
    },
}


def tune_model(
    model_name,
    pipeline,
    X,
    y,
    random_state=42,
    *,
    fold_builder=None,
    cv=3,
):
    """Ajusta hiperparametros.

    Quando fold_builder e fornecido, cada fold pode reconstruir features
    e selecao de features usando apenas o treino daquele fold.
    """
    grid = GRIDS[model_name]

    if fold_builder is None:
        search = RandomizedSearchCV(
            pipeline,
            grid,
            n_iter=4,
            scoring="r2",
            cv=KFold(n_splits=cv, shuffle=True, random_state=random_state),
            random_state=random_state,
            n_jobs=1,
            refit=True,
        )
        search.fit(X, y)
        return search.best_estimator_, search.best_params_, float(search.best_score_)

    folds = list(KFold(
        n_splits=cv,
        shuffle=True,
        random_state=random_state,
    ).split(X))

    candidates = list(ParameterSampler(
        grid,
        n_iter=min(4, max(1, np.prod([len(v) for v in grid.values()]))),
        random_state=random_state,
    ))

    best_score = -np.inf
    best_params = candidates[0]

    for params in candidates:
        scores = []
        for train_idx, val_idx in folds:
            X_fold, X_val, y_fold, y_val = fold_builder(train_idx, val_idx)
            if len(X_fold) < 2 or len(X_val) == 0:
                continue

            estimator = clone(pipeline)
            estimator.set_params(**params)
            estimator.fit(X_fold, y_fold)
            prediction = estimator.predict(X_val)
            scores.append(r2_score(y_val, prediction))

        score = float(np.mean(scores)) if scores else -np.inf
        if score > best_score:
            best_score = score
            best_params = params

    final_model = clone(pipeline)
    final_model.set_params(**best_params)
    final_model.fit(X, y)

    return final_model, best_params, float(best_score)
