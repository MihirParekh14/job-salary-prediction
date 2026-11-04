"""Load job postings and put every salary on the same scale.

The hard part of this dataset is not the modelling. It is that a "salary"
column contains an hourly rate on one row and an annual package on the next,
and a separate column says which. Get that wrong and every number downstream
is quietly meaningless.

Column names differ between copies of this dataset, so each field is resolved
against a list of candidates rather than a single hard-coded key -- the same
reason the names are listed here as aliases instead of being assumed.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

# --- Column resolution -----------------------------------------------------

FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "job_id": ("job_id", "id"),
    "title": ("title", "job_title", "position"),
    "company": ("company_name", "company", "company_id"),
    "location": ("location", "job_location", "city"),
    "min_salary": ("min_salary", "salary_min", "min_pay"),
    "max_salary": ("max_salary", "salary_max", "max_pay"),
    "med_salary": ("med_salary", "median_salary", "salary", "normalized_salary"),
    "pay_period": ("pay_period", "salary_period", "period"),
    "currency": ("currency", "salary_currency"),
    "compensation_type": ("compensation_type", "comp_type"),
    "work_type": ("formatted_work_type", "work_type", "employment_type"),
    "experience_level": ("formatted_experience_level", "experience_level", "seniority"),
    "remote": ("remote_allowed", "is_remote", "remote"),
    "views": ("views",),
    "applies": ("applies", "applications"),
}

REQUIRED = ("title",)


def resolve_columns(frame: pd.DataFrame) -> dict[str, str]:
    """Map our field names onto whichever columns this copy actually has."""
    lowered = {str(column).strip().lower(): column for column in frame.columns}

    found = {}
    for field, aliases in FIELD_ALIASES.items():
        for alias in aliases:
            if alias.lower() in lowered:
                found[field] = lowered[alias.lower()]
                break
    return found


def describe_schema(frame: pd.DataFrame) -> str:
    """A readable summary of the columns, for checking a new data source."""
    found = resolve_columns(frame)
    missing = [field for field in FIELD_ALIASES if field not in found]

    lines = [f"{len(frame):,} rows, {len(frame.columns)} columns", "", "Recognised:"]
    for field, column in found.items():
        filled = frame[column].notna().mean() * 100
        lines.append(f"  {field:<18} <- {column:<32} {filled:5.1f}% filled")

    if missing:
        lines += ["", "Not found (the pipeline will work around these):"]
        lines += [f"  {field}" for field in missing]

    unused = sorted(set(frame.columns) - set(found.values()))
    if unused:
        lines += ["", f"Unused columns: {', '.join(map(str, unused[:25]))}"]
    return "\n".join(lines)


def standardise(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a frame with our field names, keeping only what was found."""
    found = resolve_columns(frame)

    missing_required = [field for field in REQUIRED if field not in found]
    if missing_required:
        raise ValueError(
            f"could not find a column for {missing_required} in "
            f"{list(frame.columns)[:15]}. Run `python -m src.prepare --inspect` "
            f"to see what this dataset actually has."
        )

    out = pd.DataFrame({field: frame[column] for field, column in found.items()})
    for field in FIELD_ALIASES:
        if field not in out.columns:
            out[field] = np.nan
    return out


# --- Pay period normalisation ----------------------------------------------


def period_factor(period: object) -> float:
    """How many of ``period`` fit in a year. NaN when the period is unknown.

    Returning NaN rather than guessing 1.0 matters. Defaulting an unrecognised
    period to "yearly" would silently treat an hourly rate as an annual salary,
    which is exactly the error this module exists to prevent.
    """
    if period is None or (isinstance(period, float) and np.isnan(period)):
        return float("nan")
    return config.PERIODS_PER_YEAR.get(str(period).strip().upper(), float("nan"))


def annualise(amount: object, period: object) -> float:
    """Convert one pay figure to an annual equivalent."""
    try:
        value = float(amount)
    except (TypeError, ValueError):
        return float("nan")
    if np.isnan(value) or value <= 0:
        return float("nan")

    factor = period_factor(period)
    if np.isnan(factor):
        return float("nan")
    return value * factor


def midpoint(row: pd.Series) -> float:
    """The salary figure to use for a posting.

    Preference order: the stated median, then the midpoint of a min-max range,
    then whichever single bound exists. A posting advertising "$80,000 to
    $120,000" is one observation at $100,000, not two observations -- counting
    both ends would double-weight every range and pull the model toward
    whichever postings happened to publish one.
    """
    med = row.get("med_salary")
    if pd.notna(med) and float(med) > 0:
        return float(med)

    low, high = row.get("min_salary"), row.get("max_salary")
    low = float(low) if pd.notna(low) and float(low) > 0 else np.nan
    high = float(high) if pd.notna(high) and float(high) > 0 else np.nan

    if not np.isnan(low) and not np.isnan(high):
        return (low + high) / 2
    if not np.isnan(low):
        return low
    if not np.isnan(high):
        return high
    return float("nan")


def add_annual_salary(frame: pd.DataFrame) -> pd.DataFrame:
    """Add an ``annual_salary`` column, NaN where it cannot be determined."""
    out = frame.copy()
    raw = out.apply(midpoint, axis=1)
    out["annual_salary"] = [
        annualise(amount, period) for amount, period in zip(raw, out["pay_period"], strict=True)
    ]
    return out


def filter_currency(frame: pd.DataFrame, currency: str = config.CURRENCY) -> pd.DataFrame:
    """Keep only postings priced in ``currency``, plus those with no salary.

    Converting the rest would need an exchange rate at each posting's date,
    which this dataset does not carry. Averaging them as if they were the same
    unit is the error; dropping them and reporting the count is the fix.

    Rows with no currency recorded *and* no salary are kept, because they are
    still needed to measure how often salaries are disclosed at all.
    """
    if "currency" not in frame.columns:
        return frame.copy()

    codes = frame["currency"].astype("object").map(
        lambda value: str(value).strip().upper() if pd.notna(value) else None
    )

    # If the dataset records no currency anywhere, there is nothing to filter
    # on and dropping every row would be absurd. Filtering only happens when
    # the information exists.
    if codes.notna().sum() == 0:
        return frame.copy()

    has_salary = (
        frame["annual_salary"].notna()
        if "annual_salary" in frame.columns
        else pd.Series(False, index=frame.index)
    )
    keep = (codes == currency.upper()) | (~has_salary)
    return frame[keep].copy()


def currency_counts(frame: pd.DataFrame) -> pd.Series:
    """How many postings are priced in each currency."""
    if "currency" not in frame.columns:
        return pd.Series(dtype=int)
    return frame["currency"].dropna().astype(str).str.upper().value_counts()


def drop_implausible(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove salaries outside the plausible band.

    These are data-entry errors and pay quoted in units the dataset did not
    record. One $8,000,000 row moves a squared-error model more than a thousand
    ordinary ones, so leaving them in is not neutral.
    """
    salary = frame["annual_salary"]
    keep = salary.between(config.MIN_ANNUAL_SALARY, config.MAX_ANNUAL_SALARY)
    return frame[keep | salary.isna()].copy()


def labelled(frame: pd.DataFrame) -> pd.DataFrame:
    """Only the postings that disclose a usable salary."""
    return frame[frame["annual_salary"].notna()].copy()


# --- Location --------------------------------------------------------------

_STATE_CODES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID",
    "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS",
    "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK",
    "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV",
    "WI", "WY", "DC",
}


def state_from_location(location: object) -> str:
    """Pull a two-letter state out of a free-text location string.

    Locations arrive as "Detroit, MI", "Michigan, United States", "Remote" and
    a dozen other shapes. Anything unrecognised becomes "UNKNOWN" rather than
    being dropped, because a missing location is itself informative.
    """
    if location is None or (isinstance(location, float) and np.isnan(location)):
        return "UNKNOWN"

    text = str(location).strip()
    if not text:
        return "UNKNOWN"

    for part in reversed([piece.strip() for piece in text.split(",")]):
        upper = part.upper()
        if upper in _STATE_CODES:
            return upper
    return "UNKNOWN"


def add_state(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["state"] = out["location"].map(state_from_location)
    return out


# --- Loading and saving ----------------------------------------------------


def fetch_postings(limit: int | None = None, streaming: bool = True) -> pd.DataFrame:
    """Stream postings from the configured Hugging Face dataset.

    ``datasets`` is imported inside the function so the rest of this module --
    and its tests -- stay importable without the heavy dependency installed.
    """
    from datasets import load_dataset  # noqa: PLC0415

    dataset = load_dataset(config.HF_DATASET, split=config.HF_SPLIT, streaming=streaming)

    rows: list[dict] = []
    for record in dataset:
        rows.append(dict(record))
        if limit is not None and len(rows) >= limit:
            break
    return pd.DataFrame(rows)


def load_csv(path: str | Path) -> pd.DataFrame:
    """Read postings from a local CSV, for a dataset downloaded by hand."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.prepare` to download, "
            f"or pass --csv with a file you already have."
        )
    return pd.read_csv(path, low_memory=False)


def save(frame: pd.DataFrame, path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return len(frame)


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    """Run the whole cleaning path over a raw frame."""
    out = standardise(frame)
    out = add_annual_salary(out)
    out = filter_currency(out)
    out = drop_implausible(out)
    return add_state(out)


def disclosure_summary(frame: pd.DataFrame, by: str = "state", min_rows: int = 200) -> pd.DataFrame:
    """How often postings disclose a salary, grouped by something.

    This is not a modelling step. It is the check that decides whether the
    model built on disclosed salaries can be said to describe the job market,
    or only the part of it that is legally required to show its numbers.
    """
    out = frame.copy()
    out["disclosed"] = out["annual_salary"].notna()

    grouped = out.groupby(by).agg(
        postings=("disclosed", "size"),
        disclosed=("disclosed", "sum"),
        median_salary=("annual_salary", "median"),
    )
    grouped["disclosure_rate"] = grouped["disclosed"] / grouped["postings"]
    grouped = grouped[grouped["postings"] >= min_rows]
    return grouped.sort_values("disclosure_rate", ascending=False)


def summarise(frames: Iterable[tuple[str, pd.DataFrame]]) -> str:
    """A short before-and-after report of how many rows survived each step."""
    return "\n".join(f"  {name:<34} {len(frame):>8,}" for name, frame in frames)
