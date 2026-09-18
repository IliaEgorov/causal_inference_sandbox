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