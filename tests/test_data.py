"""Tests for loading postings and putting every salary on the same scale.

Most of the attention goes to pay-period normalisation, because that is where
this dataset breaks silently: an hourly rate read as an annual salary produces
a number that is wrong by a factor of two thousand and looks like an ordinary
row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data import (
    add_annual_salary,
    add_state,
    annualise,
    currency_counts,
    describe_schema,
    disclosure_summary,
    drop_implausible,
    filter_currency,
    labelled,
    load_csv,
    midpoint,
    period_factor,
    prepare,
    resolve_columns,
    save,
    standardise,
    state_from_location,
)


def raw_frame(**overrides) -> pd.DataFrame:
    base = {
        "job_id": [1],
        "title": ["Data Scientist"],
        "company_name": ["Acme"],
        "location": ["Detroit, MI"],
        "min_salary": [90000.0],
        "max_salary": [110000.0],
        "med_salary": [np.nan],
        "pay_period": ["YEARLY"],
        "formatted_work_type": ["Full-time"],
        "formatted_experience_level": ["Mid-Senior level"],
        "remote_allowed": [np.nan],
    }
    base.update(overrides)
    return pd.DataFrame(base)


# --- resolving columns -----------------------------------------------------


def test_standard_column_names_are_recognised():
    found = resolve_columns(raw_frame())
    assert found["title"] == "title"
    assert found["experience_level"] == "formatted_experience_level"


def test_alternative_column_names_are_recognised():
    """Copies of this dataset name these fields differently."""
    frame = pd.DataFrame({"job_title": ["x"], "salary_min": [1], "salary_period": ["YEARLY"]})
    found = resolve_columns(frame)

    assert found["title"] == "job_title"
    assert found["min_salary"] == "salary_min"
    assert found["pay_period"] == "salary_period"


def test_column_matching_ignores_case_and_spacing():
    found = resolve_columns(pd.DataFrame({" TITLE ": ["x"], "Pay_Period": ["YEARLY"]}))
    assert "title" in found
    assert "pay_period" in found


def test_missing_fields_become_empty_columns_rather_than_crashing():
    """A copy without experience levels is still worth modelling."""
    out = standardise(pd.DataFrame({"title": ["x"], "pay_period": ["YEARLY"]}))
    assert "experience_level" in out.columns
    assert out["experience_level"].isna().all()


def test_a_frame_with_no_title_column_is_rejected_by_name():
    with pytest.raises(ValueError, match="--inspect"):
        standardise(pd.DataFrame({"salary": [1]}))


def test_describe_schema_reports_what_was_found_and_what_was_not():
    text = describe_schema(raw_frame())
    assert "title" in text
    assert "% filled" in text


# --- pay periods -----------------------------------------------------------


def test_each_known_period_converts_to_a_year():
    assert period_factor("YEARLY") == 1.0
    assert period_factor("MONTHLY") == 12.0
    assert period_factor("WEEKLY") == 52.0
    assert period_factor("HOURLY") == config.HOURS_PER_YEAR


def test_period_matching_ignores_case_and_whitespace():
    assert period_factor(" hourly ") == config.HOURS_PER_YEAR


def test_an_unknown_period_gives_nan_rather_than_assuming_yearly():
    """Defaulting to 1.0 would read $35 an hour as a $35 salary -- or worse,
    read it as $35 a year and keep it."""
    assert np.isnan(period_factor("FORTNIGHTLY"))
    assert np.isnan(period_factor(None))
    assert np.isnan(period_factor(np.nan))


def test_an_hourly_rate_becomes_a_plausible_annual_salary():
    assert annualise(50, "HOURLY") == pytest.approx(104_000)


def test_a_monthly_figure_becomes_a_plausible_annual_salary():
    assert annualise(8_000, "MONTHLY") == pytest.approx(96_000)


def test_a_yearly_figure_is_left_alone():
    assert annualise(120_000, "YEARLY") == pytest.approx(120_000)


def test_hourly_and_yearly_rows_land_in_the_same_range_after_conversion():
    """The defect this module exists to prevent, stated as a test."""
    hourly = annualise(48, "HOURLY")
    yearly = annualise(100_000, "YEARLY")

    assert abs(hourly - yearly) < 20_000


def test_a_salary_with_no_period_is_dropped_not_guessed():
    assert np.isnan(annualise(100_000, None))


def test_zero_and_negative_pay_are_rejected():
    assert np.isnan(annualise(0, "YEARLY"))
    assert np.isnan(annualise(-5, "YEARLY"))


def test_non_numeric_pay_is_rejected():
    assert np.isnan(annualise("competitive", "YEARLY"))


# --- choosing a figure from a range ----------------------------------------


def test_a_stated_median_is_preferred():
    row = pd.Series({"med_salary": 95_000, "min_salary": 80_000, "max_salary": 120_000})
    assert midpoint(row) == pytest.approx(95_000)


def test_a_range_becomes_its_midpoint_not_two_observations():
    row = pd.Series({"med_salary": np.nan, "min_salary": 80_000, "max_salary": 120_000})
    assert midpoint(row) == pytest.approx(100_000)


def test_a_one_sided_range_uses_the_bound_it_has():
    low_only = pd.Series({"med_salary": np.nan, "min_salary": 80_000, "max_salary": np.nan})
    high_only = pd.Series({"med_salary": np.nan, "min_salary": np.nan, "max_salary": 120_000})

    assert midpoint(low_only) == pytest.approx(80_000)
    assert midpoint(high_only) == pytest.approx(120_000)


def test_a_posting_with_no_salary_at_all_gives_nan():
    row = pd.Series({"med_salary": np.nan, "min_salary": np.nan, "max_salary": np.nan})
    assert np.isnan(midpoint(row))


# --- implausible values ----------------------------------------------------


def test_an_absurd_salary_is_dropped():
    """One $8,000,000 row moves a squared-error model more than a thousand
    ordinary ones."""
    frame = pd.DataFrame({"annual_salary": [95_000.0, 8_000_000.0]})
    assert list(drop_implausible(frame)["annual_salary"]) == [95_000.0]


def test_an_implausibly_small_salary_is_dropped():
    frame = pd.DataFrame({"annual_salary": [95_000.0, 400.0]})
    assert list(drop_implausible(frame)["annual_salary"]) == [95_000.0]


def test_rows_with_no_salary_survive_the_plausibility_filter():
    """They are needed to measure how often salaries are disclosed at all."""
    frame = pd.DataFrame({"annual_salary": [95_000.0, np.nan]})
    assert len(drop_implausible(frame)) == 2


def test_labelled_keeps_only_postings_that_disclose_a_salary():
    frame = pd.DataFrame({"annual_salary": [95_000.0, np.nan, 120_000.0]})
    assert len(labelled(frame)) == 2


# --- currency ----------------------------------------------------------------
#
# Mixing currencies is the same class of error as mixing pay periods: EUR
# 70,000 and USD 70,000 are not the same number, and averaging them raises
# nothing.


def test_postings_in_another_currency_are_dropped():
    frame = pd.DataFrame({
        "currency": ["USD", "EUR", "GBP"],
        "annual_salary": [100_000.0, 70_000.0, 60_000.0],
    })
    assert list(filter_currency(frame)["currency"]) == ["USD"]


def test_currency_matching_ignores_case_and_whitespace():
    frame = pd.DataFrame({"currency": [" usd "], "annual_salary": [100_000.0]})
    assert len(filter_currency(frame)) == 1


def test_undisclosed_postings_survive_the_currency_filter():
    """They carry no salary to mis-scale, and are needed for the disclosure
    rate."""
    frame = pd.DataFrame({
        "currency": [None, "EUR"],
        "annual_salary": [np.nan, 70_000.0],
    })
    assert len(filter_currency(frame)) == 1
    assert filter_currency(frame)["annual_salary"].isna().all()


def test_nothing_is_filtered_when_the_dataset_records_no_currency():
    """A copy without currency information should still run, not come back
    empty."""
    frame = pd.DataFrame({
        "currency": [None, None],
        "annual_salary": [100_000.0, 120_000.0],
    })
    assert len(filter_currency(frame)) == 2


def test_currency_counts_reports_what_is_present():
    frame = pd.DataFrame({"currency": ["USD", "USD", "EUR", None]})
    counts = currency_counts(frame)

    assert counts["USD"] == 2
    assert counts["EUR"] == 1


def test_prepare_keeps_only_usd_when_currencies_are_mixed():
    frame = pd.concat([
        raw_frame().assign(currency=["USD"]),
        raw_frame().assign(currency=["EUR"]),
    ], ignore_index=True)

    assert len(labelled(prepare(frame))) == 1


# --- location --------------------------------------------------------------


def test_a_city_state_location_gives_the_state():
    assert state_from_location("Detroit, MI") == "MI"


def test_a_three_part_location_still_gives_the_state():
    assert state_from_location("Novi, MI, United States") == "MI"


def test_an_unparseable_location_is_marked_unknown_not_dropped():
    """A missing location is itself informative."""
    assert state_from_location("Remote") == "UNKNOWN"
    assert state_from_location("") == "UNKNOWN"
    assert state_from_location(np.nan) == "UNKNOWN"


def test_state_matching_is_case_insensitive():
    assert state_from_location("detroit, mi") == "MI"


# --- the whole path --------------------------------------------------------


def test_prepare_produces_an_annual_salary_and_a_state():
    out = prepare(raw_frame())
    assert out["annual_salary"].iloc[0] == pytest.approx(100_000)
    assert out["state"].iloc[0] == "MI"


def test_prepare_handles_a_mix_of_pay_periods_in_one_frame():
    frame = pd.concat([
        raw_frame(min_salary=[45.0], max_salary=[55.0], pay_period=["HOURLY"]),
        raw_frame(min_salary=[90_000.0], max_salary=[110_000.0], pay_period=["YEARLY"]),
    ], ignore_index=True)

    salaries = prepare(frame)["annual_salary"]
    assert salaries.min() > 50_000
    assert salaries.max() < 150_000


def test_add_annual_salary_leaves_undisclosed_postings_as_nan():
    frame = raw_frame(min_salary=[np.nan], max_salary=[np.nan])
    assert np.isnan(add_annual_salary(standardise(frame))["annual_salary"].iloc[0])


def test_add_state_adds_one_state_per_row():
    out = add_state(standardise(raw_frame()))
    assert out["state"].tolist() == ["MI"]


# --- disclosure --------------------------------------------------------------


def test_disclosure_summary_reports_the_rate_per_group():
    frame = pd.DataFrame({
        "state": ["CA"] * 4 + ["TX"] * 4,
        "annual_salary": [100_000.0, 120_000.0, 110_000.0, 130_000.0,
                          90_000.0, np.nan, np.nan, np.nan],
    })
    summary = disclosure_summary(frame, min_rows=1)

    assert summary.loc["CA", "disclosure_rate"] == pytest.approx(1.0)
    assert summary.loc["TX", "disclosure_rate"] == pytest.approx(0.25)


def test_disclosure_summary_drops_groups_too_small_to_mean_anything():
    frame = pd.DataFrame({
        "state": ["CA"] * 3 + ["WY"],
        "annual_salary": [100_000.0, 120_000.0, 110_000.0, 90_000.0],
    })
    assert "WY" not in disclosure_summary(frame, min_rows=3).index


# --- saving and loading ----------------------------------------------------


def test_save_then_load_round_trips(tmp_path):
    path = tmp_path / "nested" / "prepared.csv"
    frame = prepare(raw_frame())

    assert save(frame, path) == len(frame)
    assert len(load_csv(path)) == len(frame)


def test_load_csv_raises_a_helpful_error_when_file_is_absent(tmp_path):
    with pytest.raises(FileNotFoundError, match="src.prepare"):
        load_csv(tmp_path / "nothing.csv")
