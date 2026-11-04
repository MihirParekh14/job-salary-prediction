"""Download job postings, normalise pay, and write a modelling table.

Usage:
    python -m src.prepare --inspect          # show the real schema, change nothing
    python -m src.prepare                    # download, clean, save
    python -m src.prepare --limit 20000      # a quick trial run
    python -m src.prepare --csv postings.csv # use a file downloaded by hand
"""

from __future__ import annotations

import argparse
import sys
import time

from src import config
from src.data import (
    add_annual_salary,
    add_state,
    currency_counts,
    describe_schema,
    disclosure_summary,
    drop_implausible,
    fetch_postings,
    filter_currency,
    labelled,
    load_csv,
    save,
    standardise,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare job postings for modelling.")
    parser.add_argument("--inspect", action="store_true",
                        help="Print the columns this dataset actually has, then stop.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only read this many postings. Useful for a trial run.")
    parser.add_argument("--csv", default=None,
                        help="Read from a local CSV instead of downloading.")
    parser.add_argument("--out", default=str(config.PREPARED_FILE),
                        help="Where to write the prepared table.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config.ensure_dirs()

    if args.csv:
        print(f"Reading {args.csv} ...")
        raw = load_csv(args.csv)
    else:
        print(f"Downloading from {config.HF_DATASET} ...")
        started = time.time()
        raw = fetch_postings(limit=args.limit)
        print(f"Read {len(raw):,} postings in {time.time() - started:.1f}s")

    if args.inspect:
        print("\n" + describe_schema(raw))
        print("\nNothing was written. Re-run without --inspect to build the table.")
        return

    # --- clean, one step at a time so the losses are visible ---------------

    standard = standardise(raw)
    with_salary = add_annual_salary(standard)
    one_currency = filter_currency(with_salary)
    plausible = drop_implausible(one_currency)
    final = add_state(plausible)
    usable = labelled(final)

    save(final, args.out)

    print("\nRows at each step")
    print(f"  downloaded                         {len(raw):>8,}")
    print(f"  after standardising columns        {len(standard):>8,}")
    print(f"  after keeping {config.CURRENCY} only             {len(one_currency):>8,}")
    print(f"  after dropping implausible pay     {len(plausible):>8,}")
    print(f"  with a usable annual salary        {len(usable):>8,}")

    disclosed = len(usable) / max(len(final), 1)
    print(f"\n  salary disclosure rate             {disclosed:>8.1%}")

    if usable.empty:
        print("\nNo usable salaries found. Run with --inspect to check the columns.")
        return

    salary = usable["annual_salary"]
    print("\nAnnual salary, once every pay period is normalised")
    print(f"  median                             ${salary.median():>10,.0f}")
    print(f"  25th - 75th percentile             ${salary.quantile(.25):>10,.0f} "
          f"- ${salary.quantile(.75):,.0f}")
    print(f"  min - max                          ${salary.min():>10,.0f} "
          f"- ${salary.max():,.0f}")

    currencies = currency_counts(standard)
    if len(currencies) > 1:
        print("\nCurrencies present in the raw data")
        for code, count in currencies.head(8).items():
            print(f"  {code:<16} {count:>8,}")
        print(f"\n  Only {config.CURRENCY} is kept. Converting the rest would need an")
        print("  exchange rate per posting date, which this dataset does not carry.")

    periods = standard["pay_period"].dropna().astype(str).str.upper().value_counts()
    if not periods.empty:
        print("\nPay periods present in the raw data")
        for period, count in periods.head(8).items():
            print(f"  {period:<16} {count:>8,}")
        print("\n  Treating these as one unit would be the single most damaging")
        print("  thing you could do to this dataset.")

    by_state = disclosure_summary(final)
    if not by_state.empty:
        def show(rows) -> None:
            for state, row in rows.iterrows():
                print(f"    {state:<8} {row['disclosure_rate']:>6.1%}  "
                      f"of {int(row['postings']):>6,} postings")

        print("\nSalary disclosure rate by state (states with 200+ postings)")
        if len(by_state) <= 12:
            show(by_state)
        else:
            print("  highest")
            show(by_state.head(5))
            print("  lowest")
            show(by_state.tail(5))

        spread = by_state["disclosure_rate"]
        print(f"\n  {spread.max():.0%} in {spread.idxmax()} against "
              f"{spread.min():.0%} in {spread.idxmin()}.")
        print("  Which postings show a salary is decided by where the job is.")

    print(f"\nWrote {len(final):,} rows to {args.out}")
    print("Next:  python -m src.train")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nStopped before writing. Re-run to start again.")
        sys.exit(130)
