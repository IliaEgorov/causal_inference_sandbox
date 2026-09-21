#!/usr/bin/env python3
"""Attribute a change in total ARPU to traffic cohorts.

Input is a daily (or weekly) table at the grain
date x traffic_type x traffic_type_group.  Required columns:
    date, traffic_type, traffic_type_group, players
and one of:
    revenue | deposits | arpu

Optional columns include ftd and deposits.  The script aggregates the two
windows, then uses an exact symmetric (Shapley) decomposition of

    total ARPU = sum(cohort player share * cohort ARPU).

It separates each cohort's contribution into mix (player-share) and ARPU
quality effects.  The two effects sum exactly to the observed total change.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

COHORT_COLUMNS = ["traffic_type", "traffic_type_group"]


def load_and_validate(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"date", "players", *COHORT_COLUMNS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if not ({"revenue", "deposits", "arpu"} & set(df.columns)):
        raise ValueError("Supply at least one of: revenue, deposits, arpu")

    df["date"] = pd.to_datetime(df["date"])
    df["players"] = pd.to_numeric(df["players"], errors="raise")
    if (df["players"] < 0).any():
        raise ValueError("players cannot be negative")
    for col in ("revenue", "deposits", "arpu", "ftd"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="raise")

    # Revenue is the preferred numerator. Deposits is a practical substitute.
    if "revenue" not in df:
        if "deposits" in df:
            df["revenue"] = df["deposits"]
        else:
            df["revenue"] = df["arpu"] * df["players"]
    return df


def aggregate_window(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    window = df.loc[(df.date >= start) & (df.date <= end)].copy()
    if window.empty:
        raise ValueError(f"No rows between {start} and {end}")

    aggregations = {"players": "sum", "revenue": "sum"}
    if "ftd" in window:
        aggregations["ftd"] = "sum"
    if "deposits" in window:
        aggregations["deposits"] = "sum"
    out = window.groupby(COHORT_COLUMNS, as_index=False).agg(aggregations)
    out["arpu"] = np.divide(
        out.revenue, out.players, out=np.zeros(len(out)), where=out.players.ne(0)
    )
    out["player_share"] = out.players / out.players.sum()
    if "ftd" in out:
        out["ftd_rate"] = np.divide(
            out.ftd, out.players, out=np.zeros(len(out)), where=out.players.ne(0)
        )
    if "deposits" in out and "ftd" in out:
        out["adpu"] = np.divide(
            out.deposits, out.ftd, out=np.zeros(len(out)), where=out.ftd.ne(0)
        )
    return out


def decompose_arpu(baseline: pd.DataFrame, comparison: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Exact two-factor Shapley decomposition of total-ARPU movement."""
    b = baseline.rename(columns={c: f"{c}_base" for c in baseline.columns if c not in COHORT_COLUMNS})
    c = comparison.rename(columns={c: f"{c}_comp" for c in comparison.columns if c not in COHORT_COLUMNS})
    d = b.merge(c, on=COHORT_COLUMNS, how="outer").fillna(0)

    w0, w1 = d.player_share_base, d.player_share_comp
    a0, a1 = d.arpu_base, d.arpu_comp
    # Symmetric decomposition: no arbitrary choice of 'hold mix/quality fixed'.
    d["mix_effect"] = (w1 - w0) * (a0 + a1) / 2
    d["arpu_quality_effect"] = (a1 - a0) * (w0 + w1) / 2
    d["total_effect"] = d.mix_effect + d.arpu_quality_effect
    d["effect_share"] = np.divide(
        d.total_effect, d.total_effect.sum(), out=np.zeros(len(d)), where=d.total_effect.sum() != 0
    )
    d["primary_driver"] = np.where(
        d.mix_effect.abs() >= d.arpu_quality_effect.abs(), "mix / player share", "cohort ARPU"
    )
    d = d.sort_values("total_effect", key=lambda x: x.abs(), ascending=False)

    total_base = baseline.revenue.sum() / baseline.players.sum()
    total_comp = comparison.revenue.sum() / comparison.players.sum()
    summary = {
        "baseline_arpu": total_base,
        "comparison_arpu": total_comp,
        "arpu_change": total_comp - total_base,
        "arpu_change_pct": (total_comp / total_base - 1) if total_base else np.nan,
        "mix_effect": d.mix_effect.sum(),
        "quality_effect": d.arpu_quality_effect.sum(),
        "reconciliation_error": d.total_effect.sum() - (total_comp - total_base),
    }
    return d, summary


def causal_did(df: pd.DataFrame, treated: str, intervention_date: str) -> pd.DataFrame:
    """Optional DiD estimate for one traffic_type, using other types as controls.

    This estimates association under parallel-trends assumptions; it does not
    establish causality automatically.  For credible results, include only
    untreated control cohorts and inspect their pre-period trends.
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError as exc:
        raise RuntimeError("Causal mode requires: pip install statsmodels") from exc

    panel = df.copy()
    panel["arpu"] = np.divide(panel.revenue, panel.players, out=np.zeros(len(panel)), where=panel.players.ne(0))
    panel["treated"] = (panel.traffic_type == treated).astype(int)
    panel["post"] = (panel.date >= pd.Timestamp(intervention_date)).astype(int)
    # Date fixed effects remove shocks shared by all cohorts; cohort fixed effects
    # remove stable cohort-level differences. WLS respects different cohort sizes.
    model = smf.wls(
        "arpu ~ treated:post + C(date) + C(traffic_type):C(traffic_type_group)",
        data=panel, weights=panel.players.clip(lower=1),
    ).fit(cov_type="HC1")
    term = "treated:post"
    return pd.DataFrame({
        "estimate": [model.params[term]],
        "std_error": [model.bse[term]],
        "p_value": [model.pvalues[term]],
        "ci_95_low": [model.conf_int().loc[term, 0]],
        "ci_95_high": [model.conf_int().loc[term, 1]],
        "interpretation": ["estimated incremental ARPU after intervention vs controls"],
    })


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input_csv")
    p.add_argument("--baseline", required=True, nargs=2, metavar=("START", "END"))
    p.add_argument("--comparison", required=True, nargs=2, metavar=("START", "END"))
    p.add_argument("--out-dir", default="arpu_analysis")
    p.add_argument("--did-treated", help="traffic_type to evaluate in optional DiD")
    p.add_argument("--intervention-date", help="YYYY-MM-DD; required with --did-treated")
    args = p.parse_args()
    if bool(args.did_treated) != bool(args.intervention_date):
        p.error("--did-treated and --intervention-date must be supplied together")

    df = load_and_validate(args.input_csv)
    baseline = aggregate_window(df, *args.baseline)
    comparison = aggregate_window(df, *args.comparison)
    drivers, summary = decompose_arpu(baseline, comparison)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    drivers.to_csv(out / "arpu_drivers.csv", index=False)
    pd.DataFrame([summary]).to_csv(out / "arpu_summary.csv", index=False)
    if args.did_treated:
        causal_did(df, args.did_treated, args.intervention_date).to_csv(out / "did_estimate.csv", index=False)

    print(pd.Series(summary).to_string())
    print("\nLargest drivers:\n", drivers[COHORT_COLUMNS + ["mix_effect", "arpu_quality_effect", "total_effect", "primary_driver"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
