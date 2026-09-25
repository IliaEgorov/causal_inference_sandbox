"""The smallest useful weekly MMM: linear regression on spend and confounders.

Input CSV: week, FTDs, Channel_Spend_*, optional Confounder_*.
This intentionally has no adstock, saturation, seasonality, constraints, or
hyperparameter search. Start here, then move to mmm.py feature by feature.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def fit_simple_mmm(data: pd.DataFrame, target: str = "FTDs") -> tuple[float, pd.Series, pd.DataFrame]:
    required = {"week", target}
    if missing := required - set(data.columns):
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    features = [column for column in data.columns if column.startswith("Channel_Spend_") or column.startswith("Confounder_")]
    if not any(column.startswith("Channel_Spend_") for column in features):
        raise ValueError("Add at least one Channel_Spend_* column")
    frame = data.sort_values("week").reset_index(drop=True)
    x = frame[features].apply(pd.to_numeric, errors="raise").to_numpy(float)
    y = pd.to_numeric(frame[target], errors="raise").to_numpy(float)
    coefficients = np.linalg.lstsq(np.column_stack([np.ones(len(frame)), x]), y, rcond=None)[0]
    intercept, beta = float(coefficients[0]), pd.Series(coefficients[1:], index=features, name="ftds_per_eur")
    fitted = intercept + x @ beta.to_numpy()
    result = pd.DataFrame({"week": frame["week"], target: y, "fitted_ftds": fitted, "residual": y - fitted})
    for channel in [column for column in features if column.startswith("Channel_Spend_")]:
        result[f"contribution_{channel}"] = frame[channel].to_numpy(float) * beta[channel]
    return intercept, beta, result


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit the simplest possible weekly FTD MMM")
    parser.add_argument("data", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/mmm_simple"))
    args = parser.parse_args()
    intercept, coefficients, weekly = fit_simple_mmm(pd.read_csv(args.data))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"feature": coefficients.index, "ftds_per_eur": coefficients.values}).to_csv(args.output_dir / "coefficients.csv", index=False)
    weekly.to_csv(args.output_dir / "weekly_fit.csv", index=False)
    print(f"Intercept: {intercept:.3f} FTDs")
    print(coefficients.to_string())


if __name__ == "__main__":
    main()
