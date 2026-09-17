from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

NON_MODEL_FEATURES = {
    "Unique ID", "Player", "Best Position", "Position", "Primary Position",
    "Secondary Positions", "All Positions", "Division", "Preferred Foot",
    "Expires", "Rating",
}

ECONOMIC_FEATURES = {
    "Salary", "Transfer Value", "salary_min_brl", "salary_max_brl",
    "transfer_value_min_brl", "transfer_value_max_brl",
}

EXCLUDED_FEATURES = NON_MODEL_FEATURES | ECONOMIC_FEATURES

def get_stat_features(df: pd.DataFrame, include_economic: bool = False) -> list[str]:
    excluded = NON_MODEL_FEATURES if include_economic else EXCLUDED_FEATURES
    return [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]

def get_selectable_features(X: pd.DataFrame, include_economic: bool = False) -> list[str]:
    excluded = NON_MODEL_FEATURES if include_economic else EXCLUDED_FEATURES
    return [c for c in X.columns if c not in excluded and pd.api.types.is_numeric_dtype(X[c])]

def rank_features_for_target(X_train: pd.DataFrame, y_train: pd.Series, *, min_relevance: float = 0.05, mi_neighbors: int = 3, random_state: int = 42, include_economic: bool = False) -> pd.DataFrame:
    features = get_selectable_features(X_train, include_economic=include_economic)
    if not features:
        return pd.DataFrame(columns=["Feature", "Pearson", "Spearman", "MutualInformation", "Score", "Relevant"])
    X = X_train[features].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(y_train, errors="coerce")
    valid = y.notna()
    X = X.loc[valid].replace([np.inf, -np.inf], np.nan)
    y = y.loc[valid]
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    pearson = X.corrwith(y).abs().fillna(0)
    spearman = X.corrwith(y, method="spearman").abs().fillna(0)
    try:
        mi = mutual_info_regression(X, y, n_neighbors=min(mi_neighbors, max(1, len(X)-1)), random_state=random_state) if len(X) > 2 else np.zeros(len(features))
    except Exception:
        mi = np.zeros(len(features))
    mi_s = pd.Series(mi, index=features).fillna(0)
    def norm(s):
        lo, hi = float(s.min()), float(s.max())
        return (s-lo)/(hi-lo) if hi > lo else s*0
    score = (norm(pearson) + norm(spearman) + norm(mi_s)) / 3
    result = pd.DataFrame({"Feature": features, "Pearson": pearson, "Spearman": spearman, "MutualInformation": mi_s, "Score": score})
    result["Relevant"] = (result[["Pearson", "Spearman", "MutualInformation"]] >= min_relevance).any(axis=1)
    return result.sort_values(["Relevant", "Score"], ascending=[False, False]).reset_index(drop=True)

def select_features_for_target(X_train: pd.DataFrame, y_train: pd.Series, *, correlation_threshold: float = 0.95, min_relevance: float = 0.05, max_features: int = 30, mi_neighbors: int = 3, random_state: int = 42, include_economic: bool = False) -> list[str]:
    ranking = rank_features_for_target(X_train, y_train, min_relevance=min_relevance, mi_neighbors=mi_neighbors, random_state=random_state, include_economic=include_economic)
    candidates = ranking.loc[ranking["Relevant"], "Feature"].tolist()
    if not candidates:
        candidates = ranking["Feature"].head(max_features).tolist()
    selected = []
    corr = X_train[candidates].apply(pd.to_numeric, errors="coerce").corr(method="spearman").abs()
    for feature in candidates:
        if len(selected) >= max_features:
            break
        if not selected or not any(corr.loc[feature, chosen] > correlation_threshold for chosen in selected):
            selected.append(feature)
    return selected

def select_low_correlation_features(X_train: pd.DataFrame, threshold: float = 0.95) -> list[str]:
    numeric = X_train[get_selectable_features(X_train)].select_dtypes(include="number")
    if numeric.shape[1] <= 1:
        return list(numeric.columns)
    corr = numeric.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    return [column for column in numeric.columns if not (upper[column] > threshold).any()]
