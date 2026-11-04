"""Shared settings.

Every tunable number lives here so an experiment changes one file, not five.
"""

from __future__ import annotations

from pathlib import Path

# --- Paths -----------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

POSTINGS_FILE = RAW_DIR / "postings.csv"
PREPARED_FILE = PROCESSED_DIR / "prepared.csv"

# --- Data source -----------------------------------------------------------

HF_DATASET = "xanderios/linkedin-job-postings"
HF_SPLIT = "train"

# --- Pay period normalisation ----------------------------------------------
#
# Postings quote pay in whatever unit the employer used: an hourly rate, a
# monthly figure, an annual package. They sit in the same column with a
# separate field naming the period.
#
# Mixing them is the quiet defect in this dataset. A model trained on raw
# amounts learns that $35 and $150,000 are both salaries and tries to fit a
# line through them, and nothing in the output says anything is wrong -- the
# error just comes out enormous and gets blamed on the features.
#
# A US full-time year is taken as 2,080 hours: 40 hours over 52 weeks. That is
# the standard convention, and it is written here rather than inline so it can
# be argued with.
HOURS_PER_WEEK = 40
WEEKS_PER_YEAR = 52
HOURS_PER_YEAR = HOURS_PER_WEEK * WEEKS_PER_YEAR  # 2080

PERIODS_PER_YEAR: dict[str, float] = {
    "HOURLY": HOURS_PER_YEAR,
    "DAILY": 260.0,  # 5 days a week
    "WEEKLY": float(WEEKS_PER_YEAR),
    "BIWEEKLY": 26.0,
    "SEMIMONTHLY": 24.0,
    "MONTHLY": 12.0,
    "QUARTERLY": 4.0,
    "YEARLY": 1.0,
    "ANNUAL": 1.0,
    "ONCE": 1.0,
}

# Postings quote pay in their own currency. Mixing currencies is the same
# class of error as mixing pay periods -- EUR 70,000 and USD 70,000 are not the
# same number, and nothing in the data objects if they are averaged together.
# Converting between them would need exchange rates at each posting's date,
# which the dataset does not carry, so the honest move is to keep one currency
# and say how many rows that cost.
CURRENCY = "USD"

# Salaries outside this band are dropped as data-entry errors or as pay quoted
# in a period the dataset did not record. Both exist, and a single $8,000,000
# "salary" moves a mean-squared-error model more than a thousand ordinary rows.
MIN_ANNUAL_SALARY = 10_000.0
MAX_ANNUAL_SALARY = 1_000_000.0

# --- Modelling -------------------------------------------------------------

TARGET = "annual_salary"

# The target is log-transformed before fitting. Salaries are right-skewed, so
# squared error on the raw scale is dominated by a handful of very large
# values; a $40k error on a $500k package counts the same as being wrong by a
# factor of two on a $40k one. Errors are reported back on the dollar scale,
# because "0.23 log dollars" means nothing to a reader.
LOG_TARGET = True

TEST_SIZE = 0.2
RANDOM_SEED = 42
CV_FOLDS = 5

MAX_TITLE_FEATURES = 300
RANDOM_FOREST_TREES = 200
MAX_TREE_DEPTH = 12


def ensure_dirs() -> None:
    """Create the data and report directories if they do not exist yet."""
    for directory in (RAW_DIR, PROCESSED_DIR, REPORTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
