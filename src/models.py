"""Model definitions, metrics, and the baseline everything is judged against.

The baseline is not decoration. A salary model that beats nothing is worth
nothing, and "R² = 0.42" means little until you know that predicting the
median for every posting already gets you a mean error of $28,000. Every model
here is reported against that number.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from src import config
from src.features import build_preprocessor


@dataclass(frozen=True)
class Scores:
    """Errors on the dollar scale, whatever scale the model was fitted on."""

    mae: float
    rmse: float
    r2: float
    median_absolute_error: float

    def as_row(self, name: str, baseline_mae: float | None = None) -> str:
        improvement = ""
        if baseline_mae:
            improvement = f"{(1 - self.mae / baseline_mae) * 100:>8.1f}%"
        return (f"  {name:<26} ${self.mae:>9,.0f} ${self.rmse:>10,.0f} "
                f"${self.median_absolute_error:>9,.0f} {self.r2:>7.3f} {improvement}")


def to_log(y: np.ndarray) -> np.ndarray:
    """Salaries are right-skewed, so fit on log dollars."""
    return np.log1p(np.asarray(y, dtype=float))


def from_log(y: np.ndarray) -> np.ndarray:
    """Back to dollars, so the reported error is a number people understand."""
    return np.expm1(np.asarray(y, dtype=float))


def score(y_true_dollars: np.ndarray, y_pred_dollars: np.ndarray) -> Scores:
    """Score on the dollar scale, never the log scale.

    Reporting "MAE 0.23" because the model was fitted on logs is how a result
    becomes unreadable. The fit happens in log space; the report happens in
    dollars.
    """
    true = np.asarray(y_true_dollars, dtype=float)
    pred = np.asarray(y_pred_dollars, dtype=float)

    errors = np.abs(true - pred)
    mae = float(errors.mean())
    rmse = float(np.sqrt(((true - pred) ** 2).mean()))
    median_ae = float(np.median(errors))

    total = ((true - true.mean()) ** 2).sum()
    residual = ((true - pred) ** 2).sum()
    r2 = float(1 - residual / total) if total > 0 else 0.0

    return Scores(mae=mae, rmse=rmse, r2=r2, median_absolute_error=median_ae)


def build_models(seed: int = config.RANDOM_SEED) -> dict[str, object]:
    """The estimators to compare, baseline first.

    Ridge rather than plain linear regression: one-hot encoding plus TF-IDF
    produces several hundred mostly-zero columns, and ordinary least squares on
    that is unstable. The regularisation is doing real work, not decoration.
    """
    return {
        "baseline (predict median)": DummyRegressor(strategy="median"),
        "ridge regression": Ridge(alpha=1.0, random_state=seed),
        "decision tree": DecisionTreeRegressor(
            max_depth=config.MAX_TREE_DEPTH, min_samples_leaf=20, random_state=seed,
        ),
        "random forest": RandomForestRegressor(
            n_estimators=config.RANDOM_FOREST_TREES,
            max_depth=config.MAX_TREE_DEPTH,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=seed,
        ),
    }


def build_pipeline(estimator) -> Pipeline:
    """Preprocessing and the estimator as one object.

    This is the whole point. Fitting the pipeline fits the encoder and the
    vectoriser on the training rows only, so nothing about the held-out rows
    leaks into the transformation. Doing it by hand -- vectorise everything,
    then split -- is the defect that made the original version of the churn
    project report scores it had not earned.
    """
    return Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", estimator),
    ])
