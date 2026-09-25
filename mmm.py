"""Dependency-light weekly Marketing Mix Model for FTDs.

Input columns: week, FTDs, Channel_Spend_*, Confounder_*.
The implementation uses adstock + Hill saturation, time-series validation, and
ridge regression with non-negative media coefficients.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class MMMResult:
    alpha: float
    ridge_lambda: float
    intercept: float
    coefficients: pd.Series
    fitted: np.ndarray
    contributions: pd.DataFrame
    validation_rmse: float


def adstock(spend: np.ndarray, alpha: float) -> np.ndarray:
    """Geometric carry-over. alpha=0 means no carry-over."""
    result = np.zeros(len(spend), dtype=float)
    for index, value in enumerate(spend):
        result[index] = value + (alpha * result[index - 1] if index else 0.0)
    return result


def hill(values: np.ndarray, half_saturation: float, slope: float = 1.5) -> np.ndarray:
    """Diminishing-return transform bounded between zero and one."""
    positive = np.clip(values, 0, None)
    half_saturation = max(float(half_saturation), 1e-8)
    return positive**slope / (positive**slope + half_saturation**slope)


def _time_features(weeks: pd.Series) -> pd.DataFrame:
    dates = pd.to_datetime(weeks, errors="raise")
    trend = np.arange(len(dates), dtype=float)
    week_of_year = dates.dt.isocalendar().week.astype(float).to_numpy()
    return pd.DataFrame({
        "trend": trend,
        "seasonality_sin": np.sin(2 * np.pi * week_of_year / 52.18),
        "seasonality_cos": np.cos(2 * np.pi * week_of_year / 52.18),
    }, index=weeks.index)


def build_features(data: pd.DataFrame, alpha: float, spend_prefix: str, confounder_prefix: str) -> tuple[pd.DataFrame, list[str]]:
    channels = [column for column in data.columns if column.startswith(spend_prefix)]
    confounders = [column for column in data.columns if column.startswith(confounder_prefix)]
    if not channels:
        raise ValueError(f"No channel columns found with prefix '{spend_prefix}'")
    features = _time_features(data["week"])
    for channel in channels:
        spend = pd.to_numeric(data[channel], errors="raise").to_numpy(float)
        carried = adstock(spend, alpha)
        features[channel] = hill(carried, np.quantile(carried[carried > 0], .75) if np.any(carried > 0) else 1.0)
    for confounder in confounders:
        features[confounder] = pd.to_numeric(data[confounder], errors="raise").to_numpy(float)
    return features, channels


def _fit_constrained_ridge(x: np.ndarray, y: np.ndarray, media_indices: list[int], ridge_lambda: float, iterations: int = 12_000) -> tuple[float, np.ndarray]:
    """Projected-gradient ridge: media coefficients are constrained >= 0."""
    x_mean, x_scale = x.mean(axis=0), x.std(axis=0)
    x_scale[x_scale < 1e-9] = 1.0
    xs, y_mean = (x - x_mean) / x_scale, y.mean()
    beta = np.zeros(x.shape[1])
    lipschitz = 2 * (np.linalg.norm(xs, ord=2) ** 2 / len(y) + ridge_lambda)
    step = 1 / max(lipschitz, 1e-8)
    for _ in range(iterations):
        gradient = -2 * xs.T @ (y - y_mean - xs @ beta) / len(y) + 2 * ridge_lambda * beta
        next_beta = beta - step * gradient
        next_beta[media_indices] = np.maximum(0, next_beta[media_indices])
        if np.max(np.abs(next_beta - beta)) < 1e-9:
            beta = next_beta
            break
        beta = next_beta
    original_beta = beta / x_scale
    intercept = y_mean - x_mean @ original_beta
    return float(intercept), original_beta


def _fit_candidate(features: pd.DataFrame, target: np.ndarray, channels: list[str], ridge_lambda: float, train_end: int) -> tuple[float, np.ndarray, float]:
    x = features.to_numpy(float)
    media_indices = [features.columns.get_loc(channel) for channel in channels]
    intercept, coefficients = _fit_constrained_ridge(x[:train_end], target[:train_end], media_indices, ridge_lambda)
    validation_prediction = intercept + x[train_end:] @ coefficients
    rmse = float(np.sqrt(np.mean((target[train_end:] - validation_prediction) ** 2)))
    return intercept, coefficients, rmse


def fit_mmm(data: pd.DataFrame, target_column: str = "FTDs", spend_prefix: str = "Channel_Spend_", confounder_prefix: str = "Confounder_", adstock_grid: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8), ridge_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)) -> MMMResult:
    """Fit and select a weekly MMM by holdout RMSE, preserving chronological order."""
    required = {"week", target_column}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    ordered = data.sort_values("week").reset_index(drop=True).copy()
    target = pd.to_numeric(ordered[target_column], errors="raise").to_numpy(float)
    if len(ordered) < 30:
        raise ValueError("MMM needs at least 30 weekly observations")
    train_end, best = max(20, int(len(ordered) * .8)), None
    for alpha in adstock_grid:
        features, channels = build_features(ordered, alpha, spend_prefix, confounder_prefix)
        for ridge_lambda in ridge_grid:
            intercept, coefficients, rmse = _fit_candidate(features, target, channels, ridge_lambda, train_end)
            if best is None or rmse < best[0]:
                best = (rmse, alpha, ridge_lambda, features, channels)
    assert best is not None
    validation_rmse, alpha, ridge_lambda, features, channels = best
    media_indices = [features.columns.get_loc(channel) for channel in channels]
    intercept, coefficients = _fit_constrained_ridge(features.to_numpy(float), target, media_indices, ridge_lambda)
    fitted = intercept + features.to_numpy(float) @ coefficients
    coefficient_series = pd.Series(coefficients, index=features.columns, name="coefficient")
    contributions = pd.DataFrame({"week": ordered["week"], target_column: target, "fitted_ftds": fitted})
    for channel in channels:
        contributions[f"contribution_{channel}"] = features[channel].to_numpy() * coefficient_series[channel]
        contributions[f"roi_{channel}"] = contributions[f"contribution_{channel}"] / ordered[channel].replace(0, np.nan).to_numpy(float)
    return MMMResult(alpha, ridge_lambda, intercept, coefficient_series, fitted, contributions, validation_rmse)


def save_result(result: MMMResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"feature": result.coefficients.index, "coefficient": result.coefficients.values}).to_csv(output_dir / "mmm_coefficients.csv", index=False)
    result.contributions.to_csv(output_dir / "mmm_weekly_contributions.csv", index=False)
    pd.DataFrame([{"adstock_alpha": result.alpha, "ridge_lambda": result.ridge_lambda, "intercept": result.intercept, "validation_rmse": result.validation_rmse}]).to_csv(output_dir / "mmm_model_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit a weekly FTD Marketing Mix Model")
    parser.add_argument("data", type=Path, help="CSV with week, FTDs, Channel_Spend_*, Confounder_*")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/mmm"))
    parser.add_argument("--target", default="FTDs")
    args = parser.parse_args()
    result = fit_mmm(pd.read_csv(args.data), target_column=args.target)
    save_result(result, args.output_dir)
    print(f"Selected adstock alpha={result.alpha}, ridge lambda={result.ridge_lambda}, validation RMSE={result.validation_rmse:.3f}")
    print(result.coefficients.to_string())


if __name__ == "__main__":
    main()
