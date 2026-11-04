"""Tests for features, metrics and the pipeline.

The ones worth reading are the leakage guards. A salary model that has seen the
test rows during preprocessing, or that can read the salary out of the job
description, reports an excellent score and is worthless. Neither failure
raises an error, so both get a test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src import config
from src.features import CATEGORICAL, build_preprocessor, select_features, to_remote_flag
from src.models import Scores, build_models, build_pipeline, from_log, score, to_log

rng = np.random.default_rng(0)


def postings(n: int = 400) -> pd.DataFrame:
    """A small synthetic table whose salary genuinely depends on the features."""
    levels = rng.choice(["Entry level", "Mid-Senior level", "Director"], size=n)
    states = rng.choice(["CA", "MI", "TX"], size=n)
    titles = rng.choice(
        ["data scientist", "senior data scientist", "data analyst", "staff engineer"],
        size=n,
    )

    base = np.where(levels == "Entry level", 70_000,
                    np.where(levels == "Mid-Senior level", 120_000, 180_000))
    bump = np.where(states == "CA", 20_000, 0)
    noise = rng.normal(0, 5_000, size=n)

    return pd.DataFrame({
        "title": titles,
        "experience_level": levels,
        "work_type": "Full-time",
        "state": states,
        "remote": rng.choice([0, 1], size=n),
        "annual_salary": base + bump + noise,
        "description": [f"We pay ${s:,.0f} a year." for s in base + bump + noise],
    })


# --- feature selection -----------------------------------------------------


def test_selected_features_have_the_expected_columns():
    out = select_features(postings(10))
    assert set(CATEGORICAL).issubset(out.columns)
    assert "title" in out.columns
    assert "remote" in out.columns


def test_the_job_description_is_never_a_feature():
    """Descriptions often state the salary in plain text. A model given one
    reads the answer off the page and reports a score it has not earned."""
    out = select_features(postings(10))
    assert "description" not in out.columns


def test_the_target_is_never_a_feature():
    assert config.TARGET not in select_features(postings(10)).columns


def test_a_missing_category_becomes_unknown_rather_than_nan():
    frame = postings(5)
    frame.loc[0, "experience_level"] = None
    assert select_features(frame)["experience_level"].iloc[0] == "UNKNOWN"


def test_a_missing_column_entirely_is_filled_in():
    """A copy of the dataset without experience levels should still run."""
    frame = postings(5).drop(columns=["experience_level"])
    assert (select_features(frame)["experience_level"] == "UNKNOWN").all()


def test_a_missing_title_becomes_an_empty_string_not_the_text_nan():
    """`str(np.nan)` is "nan", which TF-IDF would happily learn as a word."""
    frame = postings(5)
    frame.loc[0, "title"] = None
    assert select_features(frame)["title"].iloc[0] == ""


def test_the_remote_flag_accepts_whatever_shape_it_arrives_in():
    assert to_remote_flag(1) == 1.0
    assert to_remote_flag("true") == 1.0
    assert to_remote_flag("t") == 1.0
    assert to_remote_flag(np.nan) == 0.0
    assert to_remote_flag(None) == 0.0
    assert to_remote_flag(0) == 0.0


# --- the log transform -----------------------------------------------------


def test_log_and_back_returns_the_original():
    values = np.array([40_000.0, 95_000.0, 250_000.0])
    assert np.allclose(from_log(to_log(values)), values)


def test_the_log_transform_compresses_the_tail():
    """Which is the point: a $500k package should not outweigh ten $50k ones."""
    raw = np.array([50_000.0, 500_000.0])
    logged = to_log(raw)
    assert raw[1] / raw[0] == pytest.approx(10.0)
    assert logged[1] / logged[0] < 1.3


# --- metrics ---------------------------------------------------------------


def test_a_perfect_prediction_scores_zero_error():
    y = np.array([100_000.0, 120_000.0])
    scores = score(y, y)

    assert scores.mae == pytest.approx(0.0)
    assert scores.r2 == pytest.approx(1.0)


def test_errors_are_reported_in_dollars_not_logs():
    """Reporting "MAE 0.23" because the fit happened in log space is how a
    result stops being readable."""
    y_true = np.array([100_000.0, 100_000.0])
    y_pred = np.array([110_000.0, 90_000.0])

    assert score(y_true, y_pred).mae == pytest.approx(10_000.0)


def test_median_absolute_error_is_robust_to_one_terrible_prediction():
    """The gap between mean and median error is how much the tail is costing."""
    y_true = np.array([100_000.0] * 9 + [100_000.0])
    y_pred = np.array([100_000.0] * 9 + [1_000_000.0])

    scores = score(y_true, y_pred)
    assert scores.median_absolute_error == pytest.approx(0.0)
    assert scores.mae > 50_000


def test_r2_of_a_constant_prediction_is_about_zero():
    y_true = np.array([80_000.0, 100_000.0, 120_000.0])
    y_pred = np.full(3, y_true.mean())
    assert score(y_true, y_pred).r2 == pytest.approx(0.0, abs=1e-9)


def test_scores_render_as_a_row_with_a_comparison():
    row = Scores(mae=1.0, rmse=2.0, r2=0.5, median_absolute_error=0.5).as_row("m", 2.0)
    assert "m" in row and "50.0%" in row


# --- the pipeline ----------------------------------------------------------


def test_every_model_is_wrapped_in_a_pipeline_with_its_preprocessing():
    """Preprocessing outside the pipeline gets fitted on the test rows too, and
    the resulting score is wrong in the flattering direction."""
    for estimator in build_models().values():
        pipeline = build_pipeline(estimator)
        assert isinstance(pipeline, Pipeline)
        assert pipeline.steps[0][0] == "preprocess"


def test_the_preprocessor_learns_only_from_what_it_is_fitted_on():
    """A category seen only in the held-out rows must not create a column."""
    train = select_features(postings(50).assign(state="CA"))
    unseen = select_features(postings(5).assign(state="ZZ"))

    preprocessor = build_preprocessor()
    width = preprocessor.fit_transform(train).shape[1]

    assert preprocessor.transform(unseen).shape[1] == width


def test_a_category_absent_from_training_does_not_crash_prediction():
    frame = postings(300)
    X, y = select_features(frame), frame[config.TARGET].to_numpy()
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=0)
    X_test = X_test.assign(state="ZZ")

    pipeline = build_pipeline(build_models()["ridge regression"])
    pipeline.fit(X_train, to_log(y_train))
    assert len(pipeline.predict(X_test)) == len(X_test)


def test_the_baseline_is_included_and_is_a_constant_prediction():
    """Without it, "R² = 0.42" means nothing."""
    models = build_models()
    assert "baseline (predict median)" in models

    frame = postings(200)
    X, y = select_features(frame), frame[config.TARGET].to_numpy()

    pipeline = build_pipeline(models["baseline (predict median)"])
    pipeline.fit(X, y)
    assert len(set(np.round(pipeline.predict(X), 6))) == 1


def test_a_real_model_beats_the_baseline_on_data_with_real_signal():
    """If this ever fails, something upstream has broken the features."""
    frame = postings(600)
    X, y = select_features(frame), frame[config.TARGET].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0,
    )

    def mae_for(name):
        pipeline = build_pipeline(build_models()[name])
        pipeline.fit(X_train, to_log(y_train))
        return score(y_test, from_log(pipeline.predict(X_test))).mae

    assert mae_for("random forest") < mae_for("baseline (predict median)") * 0.6


def test_models_are_seeded_so_two_runs_agree():
    frame = postings(200)
    X, y = select_features(frame), frame[config.TARGET].to_numpy()

    def fit_predict():
        pipeline = build_pipeline(build_models(seed=7)["random forest"])
        pipeline.fit(X, to_log(y))
        return pipeline.predict(X)

    assert np.allclose(fit_predict(), fit_predict())
