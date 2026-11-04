"""Turn a prepared posting table into model inputs.

Everything here is assembled as a scikit-learn transformer, never applied to a
frame directly. That is deliberate and it is the main lesson carried over from
the churn project: a transformer fitted outside the pipeline gets fitted on all
the data, including the test rows, and the resulting score is wrong in the
flattering direction. Inside a pipeline, cross-validation refits it on each
training fold and the number means what it says.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src import config

CATEGORICAL = ["experience_level", "work_type", "state"]
TEXT = "title"
NUMERIC = ["remote"]


def to_remote_flag(value: object) -> float:
    """Normalise a remote field that arrives as 1, "true", "t", or missing."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    text = str(value).strip().lower()
    if text in {"1", "1.0", "true", "t", "yes", "y"}:
        return 1.0
    return 0.0


def select_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Pick the model's inputs and put them in a known shape.

    Note what is *not* here: the job description. It frequently contains the
    salary in plain text, so a model given the description would read the
    answer off the page and report a near-perfect score that collapses the
    moment a posting omits it. Leaving it out costs accuracy and buys a number
    that is actually about the job rather than about the advertisement.
    """
    out = pd.DataFrame(index=frame.index)

    for column in CATEGORICAL:
        out[column] = (
            frame[column].astype("object").where(frame[column].notna(), "UNKNOWN").astype(str)
            if column in frame.columns else "UNKNOWN"
        )

    out[TEXT] = (
        frame[TEXT].astype("object").where(frame[TEXT].notna(), "").astype(str)
        if TEXT in frame.columns else ""
    )

    out["remote"] = (
        frame["remote"].map(to_remote_flag) if "remote" in frame.columns else 0.0
    )
    return out


def build_preprocessor(max_title_features: int = config.MAX_TITLE_FEATURES) -> ColumnTransformer:
    """One-hot the categories, TF-IDF the title, pass the flag through.

    The title carries most of the signal a posting gives away for free --
    "senior", "staff", "intern", "principal" are seniority markers, and
    "engineer" against "analyst" separates pay bands. TF-IDF over word pairs
    picks those up without anyone hand-listing them.
    """
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
        ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
    ])

    title = TfidfVectorizer(
        max_features=max_title_features,
        ngram_range=(1, 2),
        min_df=5,
        strip_accents="unicode",
        lowercase=True,
    )

    numeric = SimpleImputer(strategy="constant", fill_value=0.0)

    return ColumnTransformer(
        transformers=[
            ("categorical", categorical, CATEGORICAL),
            ("title", title, TEXT),
            ("numeric", numeric, NUMERIC),
        ],
        remainder="drop",
    )
