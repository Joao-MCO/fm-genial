from __future__ import annotations

from dataclasses import dataclass

from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class AttributeModel:
    name: str
    estimator: object

    def __post_init__(self):
        if self.name == "Ridge":
            self.pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
                ("model", self.estimator),
            ])
        else:
            self.pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("model", self.estimator),
            ])

    def fit(self, X, y):
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        return self.pipeline.predict(X)


def build_model_candidates(alpha: float = 1.0, random_state: int = 42) -> dict[str, AttributeModel]:
    return {
        "Ridge": AttributeModel("Ridge", Ridge(alpha=alpha)),
        "RandomForest": AttributeModel(
            "RandomForest",
            RandomForestRegressor(
                n_estimators=120,
                max_depth=14,
                min_samples_leaf=4,
                max_features=0.7,
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
        "ExtraTrees": AttributeModel(
            "ExtraTrees",
            ExtraTreesRegressor(
                n_estimators=150,
                max_depth=16,
                min_samples_leaf=3,
                max_features=0.8,
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
        "HistGradientBoosting": AttributeModel(
            "HistGradientBoosting",
            HistGradientBoostingRegressor(
                max_iter=300,
                learning_rate=0.05,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                l2_regularization=1.0,
                random_state=random_state,
            ),
        ),
    }