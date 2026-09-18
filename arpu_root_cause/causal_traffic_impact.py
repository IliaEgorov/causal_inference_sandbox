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
    return panel


def estimate_did(panel: pd.DataFrame) -> tuple[pd.DataFrame, object]:
    # Date FE account for day-specific shocks; the interaction is the ATT.
    model = smf.wls("y ~ treated:post_treatment + C(date) + C(treated)", panel,
                    weights=panel.weight.clip(lower=1)).fit(cov_type="HC1")
    term = "treated:post_treatment"
    ci = model.conf_int().loc[term]