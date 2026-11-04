# Job Posting Salary Prediction

Predicting advertised salary from the parts of a job posting visible before you
apply — title, seniority, work type, location — across 33,246 LinkedIn
postings.

The modelling is ordinary. Two things are worth reading:

- **38% of disclosed salaries are quoted hourly, not yearly.** They sit in the
  same column as annual figures. Nothing in the data objects if you average
  them together.
- **Only 40% of postings disclose a salary at all, and which ones do is decided
  by state** — 72% in Colorado against 19% in Louisiana. The states at the top
  are the ones with pay transparency laws. That makes the modelling sample a
  regulated subset of the job market rather than the job market.

A ridge regression lands a typical error of **$25,229** against a
median-prediction baseline of **$41,478**, and beats a random forest — because
these features are sparse, which is a reason worth knowing rather than a result
to accept.

---

## The defect that matters: pay periods

A job posting quotes pay in whatever unit the employer used. In this dataset
those land in the same column, with a separate field naming the period:

```
min_salary   max_salary   pay_period
     45.00        55.00   HOURLY
 90000.00    110000.00    YEARLY
  8000.00      9500.00    MONTHLY
```

Train a model on the raw numbers and it learns that `$50` and `$120,000` are
both salaries, and tries to fit a line through them. Nothing errors. No warning
appears. The error just comes out enormous and gets blamed on the features.

Every figure is converted to an annual equivalent before anything else happens,
using 2,080 hours for a full-time year. An **unrecognised period produces `NaN`,
never a default of 1.0** — defaulting would read an hourly rate as an annual
salary, which is precisely the failure this exists to prevent. There is a test
asserting that hourly and yearly rows land in the same range afterwards.

A posting advertising a range is one observation at its midpoint, not two
observations. Counting both ends would double-weight every posting that
published a range.

On this data the problem is not rare. **38% of disclosed salaries are quoted
hourly**, not yearly:

```
YEARLY    8,009
HOURLY    5,036
MONTHLY     224
WEEKLY       82
```

Without conversion those 5,036 rows would enter the model about two thousand
times too small, sitting next to annual figures.

There is a second unit field, `currency`, handled the same way — €70,000 is not
$70,000, and converting would need an exchange rate at each posting's date,
which the data does not carry. In this particular copy **every posting is USD,
so that filter removes nothing**. It stays because another copy of the dataset
would need it, but it is a guard that has not fired, not a finding.

---

## What the data was hiding: disclosure is not random

Most postings list no salary at all. The obvious move is to drop them and model
the rest. That is only valid if disclosure is random, and it is not.

Several US states legally require salary ranges in job advertisements. So which
postings reveal their numbers is decided largely by **where the job is** —
which is one of the features the model uses to predict salary.

`python -m src.prepare` prints the disclosure rate per state before any model
runs. On 33,246 postings:

| state | disclosure rate | postings | | state | disclosure rate | postings |
|---|---|---|---|---|---|---|
| CO | **71.7%** | 611 | | TN | 27.0% | 538 |
| CA | **68.4%** | 3,719 | | TX | 26.5% | 2,612 |
| WA | **67.7%** | 818 | | PA | 26.1% | 993 |
| NY | **61.6%** | 1,600 | | LA | **18.7%** | 214 |

Overall, 40% of postings disclose. But that ranges from 72% to 19% depending on
the state, and the four at the top — Colorado, California, Washington, New York
— are states with pay transparency laws requiring a salary range in job
advertisements. Colorado's came first.

This was not something the project set out to measure. It fell out of counting
missing values by group, which is the cheapest check in the pipeline and the
one that decided what the model can honestly claim: not "what jobs pay", but
"what jobs pay in the part of the market that is required to say".

---

## Results

13,248 postings with a usable salary, held-out 20% split, errors in dollars:

| model | MAE | RMSE | median AE | R² | vs baseline |
|---|---|---|---|---|---|
| baseline (predict median) | $41,478 | $56,487 | $33,620 | −0.05 | — |
| **ridge regression** | **$25,229** | **$38,586** | **$16,626** | **0.509** | **39.2%** |
| random forest | $28,533 | $42,077 | $19,974 | 0.416 | 31.2% |
| decision tree | $30,172 | $44,306 | $21,200 | 0.352 | 27.3% |

**The linear model beat the random forest**, which is unusual on tabular data
and has a cause worth stating. These features are mostly sparse: one-hot
categories plus 300 TF-IDF columns from the job title. Trees split one feature
at a time and do poorly on wide sparse matrices, where a regularised linear
model does well. Reaching for the most powerful model available would have been
the wrong call here, and the comparison is what showed it.

**Mean error is $25,229; median error is $16,626.** Half the predictions land
within $16.6k. The mean is dragged up by a minority of badly wrong ones, and
the likely reason is visible in the pay-period table above: hourly roles and
salaried roles occupy different pay worlds, and a single model has to cover
both. Splitting them is the obvious next experiment.

**What this says overall:** title, seniority, work type and state explain about
half the variation in advertised salary. The rest needs things a posting does
not reveal before you apply — the company, the real scope of the role, the
industry. An R² of 0.51 on four weak features is a reasonable result, and
claiming more would require pretending the other half does not exist.

The baseline is not decoration. "R² = 0.42" means nothing until you know what
predicting the median for every posting already achieves. Every model is
reported against it.

Both mean and median absolute error are shown. The gap between them is how much
a small number of badly wrong predictions is costing, which a single averaged
number hides.

---

## Quick start

```bash
git clone https://github.com/MihirParekh14/job-salary-prediction.git
cd job-salary-prediction

py -3.11 -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS and Linux

pip install -r requirements-dev.txt
```

Look at the data before trusting any assumption about it:

```bash
python -m src.prepare --inspect
```

That prints the columns the dataset actually has, how full each one is, and
which of them this pipeline recognised. It writes nothing.

Then build the table and train:

```bash
python -m src.prepare
python -m src.train
```

Useful variations:

```bash
python -m src.prepare --limit 20000       # a quick trial run
python -m src.prepare --csv postings.csv  # a file downloaded by hand
python -m src.train --no-log-target       # fit on raw dollars, to see the difference
```

Output lands in `data/`, which is gitignored.

---

## Design notes

**Column names are resolved, not assumed.** Copies of this dataset name fields
differently — `title` or `job_title`, `pay_period` or `salary_period`. Each
field is looked up against a list of candidates, and a missing one becomes an
empty column rather than a crash. `--inspect` exists for when that is not
enough.

**The target is fitted on a log scale and reported in dollars.** Salaries are
right-skewed, so squared error on the raw scale is dominated by a handful of
very large values: being $40,000 out on a $500,000 package would count the same
as being wrong by a factor of two on a $40,000 one. The fit happens in log
space; the report happens in dollars, because "MAE 0.23" is not a readable
result.

**The job description is deliberately not a feature.** Descriptions frequently
state the salary in plain text. A model given one would read the answer off the
page and report a near-perfect score that collapses on any posting that omits
it. Leaving it out costs accuracy and buys a number that is about the job
rather than about the advertisement.

**Everything is inside a scikit-learn `Pipeline`.** The encoder and the TF-IDF
vectoriser are fitted on the training rows only. Fitting them on the full frame
before splitting — the natural way to write it — lets the held-out rows
influence the transformation, and the resulting score is wrong in the
flattering direction. That specific mistake is what the
[churn project](https://github.com/MihirParekh14/telco-churn-prediction) was
rebuilt to fix; this one was built not to have it.

**Implausible salaries are dropped.** One `$8,000,000` row moves a squared-error
model more than a thousand ordinary ones, and these are data-entry errors
rather than signal.

---

## Testing

```bash
pytest --cov=src --cov-report=term-missing
ruff check src tests
```

62 tests. CI runs both on Python 3.10, 3.11 and 3.12 on every push.

No test downloads anything or touches the network — every test builds its own
frame, so the suite runs anywhere in a few seconds.

The attention goes to the failures that do not raise an error:

- **Pay-period normalisation.** That an unknown period yields `NaN` instead of
  defaulting, that an hourly and a yearly row end up in the same range, that a
  range becomes one midpoint rather than two observations.
- **Currency.** That non-USD postings are dropped, that undisclosed postings
  survive the filter because they carry no salary to mis-scale, and that a copy
  of the dataset with no currency column at all still runs rather than coming
  back empty.
- **Leakage.** That the description is never a feature, that the target is
  never a feature, that the preprocessor fitted on training rows produces the
  same width on unseen categories, and that a category absent from training
  does not crash prediction.
- **The baseline.** That it exists, that it predicts a constant, and that a
  real model beats it on data with real signal — so a broken feature pipeline
  fails the suite rather than quietly producing a worse number.

---

## Project layout

```
src/
  config.py      all tunable values in one place
  data.py        column resolution, pay-period and currency normalisation,
                 disclosure analysis
  features.py    feature selection and the preprocessing transformer
  models.py      estimators, metrics, the baseline
  prepare.py     command line: download, clean, inspect
  train.py       command line: compare models
tests/           pytest suite, no network required
data/            downloaded and generated files (gitignored)
```

---

## Licence

MIT
