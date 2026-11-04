"""Compare models on a held-out split and report errors in dollars.

Usage:
    python -m src.train
    python -m src.train --data data/processed/prepared.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config
from src.data import labelled
from src.features import select_features
from src.models import build_models, build_pipeline, from_log, score, to_log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and compare salary models.")
    parser.add_argument("--data", default=str(config.PREPARED_FILE),
                        help="The table written by `python -m src.prepare`.")
    parser.add_argument("--test-size", type=float, default=config.TEST_SIZE)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--no-log-target", action="store_true",
                        help="Fit on raw dollars instead of log dollars, to see the difference.")
    return parser.parse_args()


def load_prepared(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.prepare` first."
        )
    return pd.read_csv(path, low_memory=False)


def main() -> None:
    args = parse_args()
    config.ensure_dirs()

    frame = labelled(load_prepared(args.data))
    if len(frame) < 100:
        raise SystemExit(
            f"Only {len(frame)} postings have a usable salary. "
            f"Run `python -m src.prepare --inspect` to check the columns."
        )

    X = select_features(frame)
    y = frame[config.TARGET].to_numpy(dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed,
    )

    use_log = config.LOG_TARGET and not args.no_log_target

    print(f"Postings with a disclosed salary   {len(frame):>8,}")
    print(f"  training rows                    {len(X_train):>8,}")
    print(f"  held-out rows                    {len(X_test):>8,}")
    print(f"  target                           {'log dollars' if use_log else 'raw dollars'}")
    print(f"  median salary (training)         ${np.median(y_train):>9,.0f}\n")

    print("  model                              MAE        RMSE   median AE      R²   vs base")
    print("  " + "-" * 82)

    baseline_mae = None
    results = {}

    for name, estimator in build_models(seed=args.seed).items():
        started = time.time()
        pipeline = build_pipeline(estimator)

        pipeline.fit(X_train, to_log(y_train) if use_log else y_train)
        predicted = pipeline.predict(X_test)
        if use_log:
            predicted = from_log(predicted)

        scores = score(y_test, predicted)
        results[name] = scores
        if baseline_mae is None:
            baseline_mae = scores.mae

        print(scores.as_row(name, baseline_mae) + f"  ({time.time() - started:.1f}s)")

    best = min(results.items(), key=lambda item: item[1].mae)
    print("\nBest by mean absolute error:", best[0])
    print(f"  typical error  ${best[1].mae:,.0f}   "
          f"baseline  ${baseline_mae:,.0f}   "
          f"improvement  {(1 - best[1].mae / baseline_mae) * 100:.1f}%")

    print("\nRead the median absolute error as well as the mean. The gap between")
    print("them is how much a small number of very wrong predictions is costing.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nStopped.")
        sys.exit(130)
