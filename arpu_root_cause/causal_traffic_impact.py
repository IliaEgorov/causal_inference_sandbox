#!/usr/bin/env python3
"""Estimate the causal impact of a known change on a selected traffic cohort.

Design: Difference-in-Differences (DiD), implemented with CausalPy plus a
weighted fixed-effect regression for machine-readable estimates.  It compares
the change in an outcome for treated traffic against a control group that was
not affected by the change.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

VALID_OUTCOMES = {"arpu", "ftd_rate", "adpu"}


def parse_filters(items: list[str]) -> dict[str, str]:
    """Turn repeated `column=value` arguments into a cohort selector."""
    result = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Filter must have form column=value: {item}")
        key, value = item.split("=", 1)
        if key not in {"traffic_type", "traffic_type_group"}:
            raise ValueError("Filters are limited to traffic_type and traffic_type_group")
        result[key] = value
    return result


def selector(df: pd.DataFrame, filters: dict[str, str]) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for column, value in filters.items():
        mask &= df[column].astype(str).eq(value)
    return mask


def build_panel(raw: pd.DataFrame, treated_filters: dict[str, str], outcome: str, date: str) -> pd.DataFrame:
    required = {"date", "traffic_type", "traffic_type_group", "players"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    raw = raw.copy()
    raw["date"] = pd.to_datetime(raw.date)
    raw["players"] = pd.to_numeric(raw.players, errors="raise")
    if outcome == "arpu":
        if "revenue" not in raw:
            if "deposits" not in raw:
                raise ValueError("ARPU requires revenue (or deposits as a proxy)")
            raw["revenue"] = raw["deposits"]
        numerator = "revenue"
    elif outcome == "ftd_rate":
        if "ftd" not in raw:
            raise ValueError("ftd_rate requires ftd")
        numerator = "ftd"
    else:
        if not {"deposits", "ftd"} <= set(raw):
            raise ValueError("adpu requires deposits and ftd")
        # ADPU weights by FTD, not players.
        raw["analysis_weight"] = pd.to_numeric(raw.ftd, errors="raise")
        raw["outcome_numerator"] = pd.to_numeric(raw.deposits, errors="raise")
        denominator = "analysis_weight"
        return aggregate_groups(raw, treated_filters, denominator, "outcome_numerator", date, outcome)

    raw["analysis_weight"] = raw.players
    raw["outcome_numerator"] = pd.to_numeric(raw[numerator], errors="raise")
    return aggregate_groups(raw, treated_filters, "analysis_weight", "outcome_numerator", date, outcome)


def aggregate_groups(raw: pd.DataFrame, filters: dict[str, str], denominator: str, numerator: str, intervention: str, outcome: str) -> pd.DataFrame:
    raw["treated"] = selector(raw, filters).astype(int)
    if raw.treated.sum() == 0 or raw.treated.sum() == len(raw):
        raise ValueError("Treated selector must match some, but not all, rows; a control group is required")
    panel = raw.groupby(["date", "treated"], as_index=False).agg(
        numerator=(numerator, "sum"), weight=(denominator, "sum")
    )
    panel["y"] = np.divide(panel.numerator, panel.weight, out=np.zeros(len(panel)), where=panel.weight.ne(0))
    panel["post_treatment"] = (panel.date >= pd.Timestamp(intervention)).astype(int)
    panel["group"] = panel.treated.map({1: "treated", 0: "control"})
    panel["outcome"] = outcome
    if panel.groupby("treated").date.nunique().min() < 14:
        raise ValueError("Use at least 14 observations in each group; 28+ pre-period days are preferable")
    days_by_period = panel.groupby(["treated", "post_treatment"]).date.nunique()
    if len(days_by_period) != 4 or days_by_period.min() < 7:
        raise ValueError("Each treatment/control group needs at least 7 dates before and after the change")
    return panel


def estimate_did(panel: pd.DataFrame) -> tuple[pd.DataFrame, object]:
    # Date FE account for day-specific shocks; the interaction is the ATT.
    model = smf.wls("y ~ treated:post_treatment + C(date) + C(treated)", panel,
                    weights=panel.weight.clip(lower=1)).fit(cov_type="HC1")
    term = "treated:post_treatment"
    ci = model.conf_int().loc[term]
    effect = model.params[term]
    baseline_treated = panel.query("treated == 1 and post_treatment == 0").y.mean()
    result = pd.DataFrame([{
        "estimand": "ATT: incremental outcome for treated traffic after change",
        "estimate": effect,
        "ci_95_low": ci.iloc[0], "ci_95_high": ci.iloc[1],
        "p_value": model.pvalues[term],
        "baseline_treated_outcome": baseline_treated,
        "effect_vs_baseline_pct": effect / baseline_treated if baseline_treated else np.nan,
        "n_days": panel.date.nunique(),
    }])
    return result, model


def pretrend_check(panel: pd.DataFrame, intervention: str) -> pd.DataFrame:
    """Simple, visible check: pre-period treated-control gap must not trend."""
    pre = panel[panel.date < pd.Timestamp(intervention)].copy()
    gaps = pre.pivot(index="date", columns="treated", values="y").dropna()
    gaps.columns = ["control", "treated"]
    gaps["gap"] = gaps.treated - gaps.control
    gaps["t"] = np.arange(len(gaps))
    fit = smf.ols("gap ~ t", gaps).fit(cov_type="HC1")
    return pd.DataFrame([{
        "pre_period_days": len(gaps),
        "gap_trend_per_day": fit.params.get("t", np.nan),
        "gap_trend_p_value": fit.pvalues.get("t", np.nan),
        "interpretation": "High p-value is not proof, but supports the parallel-trends assumption.",
    }])


def save_plot(panel: pd.DataFrame, intervention: str, output: Path) -> None:
    daily = panel.pivot(index="date", columns="group", values="y").sort_index()
    ax = daily.plot(figsize=(11, 5), marker="o", alpha=.75)
    ax.axvline(pd.Timestamp(intervention), color="black", linestyle="--", label="change date")
    ax.set(title="Outcome: treated traffic vs control", xlabel="date", ylabel=panel.outcome.iloc[0])
    ax.legend()
    ax.figure.tight_layout()
    ax.figure.savefig(output, dpi=160)
    plt.close(ax.figure)


def try_causalpy(panel: pd.DataFrame, output: Path) -> str:
    """Run CausalPy's native DiD too; our CSV estimate stays version-stable."""
    try:
        import causalpy as cp
        result = cp.DifferenceInDifferences(
            panel, formula="y ~ 1 + group*post_treatment",
            time_variable_name="date", group_variable_name="group",
        )
        fig, _ = result.plot()
        fig.savefig(output / "causalpy_did_plot.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        return "CausalPy DiD executed; see causalpy_did_plot.png"
    except ImportError:
        return "CausalPy not installed; used statsmodels DiD. Install causalpy for its native diagnostics."
    except Exception as exc:  # Keep a version-specific plotting failure non-fatal.
        return f"CausalPy did not complete ({type(exc).__name__}: {exc}); statsmodels DiD still completed."


def main() -> None:
    parser = argparse.ArgumentParser(description="Causal impact of a known traffic intervention")
    parser.add_argument("input_csv")
    parser.add_argument("--change-date", required=True, help="First date on which the change was live, YYYY-MM-DD")
    parser.add_argument("--treated", required=True, nargs="+", metavar="COLUMN=VALUE",
                        help="One or two selectors, e.g. traffic_type=media traffic_type_group=ppc")
    parser.add_argument("--outcome", choices=VALID_OUTCOMES, default="arpu")
    parser.add_argument("--out-dir", default="causal_impact")
    args = parser.parse_args()

    raw = pd.read_csv(args.input_csv)
    treated = parse_filters(args.treated)
    panel = build_panel(raw, treated, args.outcome, args.change_date)
    estimate, _ = estimate_did(panel)
    pretrend = pretrend_check(panel, args.change_date)

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out / "did_daily_panel.csv", index=False)
    estimate.to_csv(out / "did_effect.csv", index=False)
    pretrend.to_csv(out / "pretrend_check.csv", index=False)
    save_plot(panel, args.change_date, out / "treated_vs_control.png")
    status = try_causalpy(panel, out)
    (out / "run_metadata.json").write_text(json.dumps({
        "change_date": args.change_date, "treated_selector": treated,
        "outcome": args.outcome, "causalpy_status": status,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(estimate.to_string(index=False)); print("\n" + status)


if __name__ == "__main__":
    main()
