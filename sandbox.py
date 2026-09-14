"""Preparation, controlled treatment simulation, and baseline ATE estimators."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW_DATA = ROOT / "data" / "Transaction-Report_20251114-expanded.csv.xz"


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30, 30)))


def _ggr(events: pd.DataFrame) -> float:
    gaming = events[events["category"].eq("Gaming")]
    return float(-gaming["amount"].sum())


def _stratum(session_id: int, name: str, values: tuple[str, ...]) -> str:
    index = int(hashlib.sha256(f"{name}:{session_id}".encode()).hexdigest(), 16) % len(values)
    return values[index]


def prepare_session_units(raw_path: Path = RAW_DATA) -> pd.DataFrame:
    """Split each real session into pre-treatment features and a post-period outcome."""
    events = pd.read_csv(raw_path, compression="xz", low_memory=False, parse_dates=["datetime"]).sort_values("datetime")
    rows: list[dict[str, object]] = []
    for session_id, session in events.groupby("session_id", sort=True):
        session = session.reset_index(drop=True)
        split = max(3, int(len(session) * 0.40))
        if len(session) < 10 or split >= len(session):
            continue
        pre, post = session.iloc[:split], session.iloc[split:]
        start, end = session["datetime"].iloc[0], session["datetime"].iloc[-1]
        stake = -pre.loc[pre["type"].eq("Stake"), "amount"].sum()
        rows.append({
            "unit_id": f"session_{session_id}", 
            "source_session_id": int(session_id),
            "session_start": start, "pre_event_count": len(pre), "pre_stake_eur": float(stake),
            "pre_ggr_eur": _ggr(pre), "pre_deposit_eur": float(pre.loc[pre["type"].eq("Deposit"), "amount"].sum()),
            "tenure_days": int((start - events["datetime"].min()).total_seconds() // 86_400),
            "session_minutes": max(1.0, (end - start).total_seconds() / 60),
            "start_hour": int(start.hour), "month": int(start.month),
            "is_weekend": int(start.dayofweek >= 5), "natural_future_ggr_eur": _ggr(post),
            "country_synthetic": _stratum(int(session_id), "country", ("DE", "GB", "CA", "FI")),
            "channel_synthetic": _stratum(int(session_id), "channel", ("affiliate", "paid_search", "organic", "direct")),
        })
    units = pd.DataFrame(rows)
    if len(units) < 50:
        raise ValueError("Too few usable sessions to create a causal sandbox")
    return units


def _standardize(frame: pd.DataFrame, columns: list[str]) -> dict[str, np.ndarray]:
    return {column: (frame[column].to_numpy(float) - frame[column].mean()) / (frame[column].std() + 1e-8) for column in columns}


def simulate_campaign(units: pd.DataFrame, level: str, seed: int = 2026) -> pd.DataFrame:
    """Add hidden assignment mechanism and known potential-outcome effect."""
    if level not in {"01_randomized", "02_observable_confounding", "03_nonlinear_confounding", "04_heterogeneous_effect"}:
        raise ValueError(f"Unknown level: {level}")
    
    result, rng = units.copy(), np.random.default_rng(seed) # init
    features = _standardize(result, ["pre_ggr_eur", "pre_stake_eur", "pre_event_count", "tenure_days"]) # standardize ggr, stake, event_cnt, tenure

    high_value = features["pre_ggr_eur"] + 0.5 * features["pre_stake_eur"] # ?
    channel = result["channel_synthetic"].map({"affiliate": .35, "paid_search": .15, "organic": -.10, "direct": -.20}).to_numpy()
    country = result["country_synthetic"].map({"DE": .15, "GB": .05, "CA": -.05, "FI": -.15}).to_numpy()

    if level == "01_randomized":
        probability, effect = np.full(len(result), .5), np.full(len(result), 10.0)
    else:
        score = -0.35 + .75 * high_value + .35 * features["pre_event_count"] + channel + country
        if level in {"03_nonlinear_confounding", "04_heterogeneous_effect"}:
            score += .45 * high_value**2 + .40 * features["pre_event_count"] * features["tenure_days"]
        probability = sigmoid(score)
        if level == "04_heterogeneous_effect":
            effect = np.select([high_value >= np.quantile(high_value, .75), high_value <= np.quantile(high_value, .25)], [30.0, -2.0], default=10.0)
        else:
            effect = np.full(len(result), 8.0)
    treatment = rng.binomial(1, probability)
    result["treatment_bonus"] = treatment
    result["true_propensity_hidden"] = probability
    result["true_individual_effect_eur"] = effect
    result["future_ggr_eur"] = result["natural_future_ggr_eur"] + treatment * effect
    result["true_ate_eur"] = float(np.mean(effect))
    return result


def design_matrix(frame: pd.DataFrame) -> np.ndarray:
    numeric = frame[["pre_ggr_eur", "pre_stake_eur", "pre_event_count", "tenure_days", "session_minutes", "start_hour", "month", "is_weekend"]].to_numpy(float)
    numeric = (numeric - numeric.mean(axis=0)) / (numeric.std(axis=0) + 1e-8)
    categorical = pd.get_dummies(frame[["country_synthetic", "channel_synthetic"]], dtype=float, drop_first=True).to_numpy()
    return np.column_stack([np.ones(len(frame)), numeric, categorical])


def _ols(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _fit_logistic(x: np.ndarray, treatment: np.ndarray, iterations: int = 4_000, learning_rate: float = .05) -> np.ndarray:
    beta = np.zeros(x.shape[1])
    for _ in range(iterations):
        prediction = sigmoid(x @ beta)
        gradient = x.T @ (prediction - treatment) / len(treatment)
        beta -= learning_rate * gradient
    return beta


def _logistic(x: np.ndarray, treatment: np.ndarray) -> np.ndarray:
    return np.clip(sigmoid(x @ _fit_logistic(x, treatment)), .03, .97)


def _predict_ols(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    return test_x @ _ols(train_x, train_y)


def dml_ate(x: np.ndarray, treatment: np.ndarray, outcome: np.ndarray, folds: int = 5) -> float:
    """Cross-fitted partially-linear DML estimator without using ground truth."""
    order = np.random.default_rng(41).permutation(len(outcome))
    fold_ids = np.empty(len(outcome), dtype=int)
    fold_ids[order] = np.arange(len(outcome)) % folds
    outcome_hat, propensity = np.empty(len(outcome)), np.empty(len(outcome))
    for fold in range(folds):
        test, train = fold_ids == fold, fold_ids != fold
        outcome_hat[test] = _predict_ols(x[train], outcome[train], x[test])
        propensity[test] = np.clip(sigmoid(x[test] @ _fit_logistic(x[train], treatment[train])), .03, .97)
    treatment_residual, outcome_residual = treatment - propensity, outcome - outcome_hat
    return float(np.sum(treatment_residual * outcome_residual) / np.sum(treatment_residual**2))


def meta_learners(x: np.ndarray, treatment: np.ndarray, outcome: np.ndarray) -> dict[str, np.ndarray]:
    """Return estimated CATE for standard linear T/S/X/DR learners."""
    treated, control = treatment == 1, treatment == 0
    mu1 = _predict_ols(x[treated], outcome[treated], x)
    mu0 = _predict_ols(x[control], outcome[control], x)
    t_learner = mu1 - mu0
    s_design = np.column_stack([x, treatment, x * treatment[:, None]])
    beta_s = _ols(s_design, outcome)
    s_learner = np.column_stack([x, np.ones(len(x)), x]) @ beta_s - np.column_stack([x, np.zeros(len(x)), np.zeros_like(x)]) @ beta_s
    d1, d0 = outcome[treated] - mu0[treated], mu1[control] - outcome[control]
    tau1 = _predict_ols(x[treated], d1, x)
    tau0 = _predict_ols(x[control], d0, x)
    propensity = _logistic(x, treatment)
    x_learner = propensity * tau0 + (1 - propensity) * tau1
    pseudo_outcome = mu1 - mu0 + treatment / propensity * (outcome - mu1) - (1 - treatment) / (1 - propensity) * (outcome - mu0)
    dr_learner = _predict_ols(x, pseudo_outcome, x)
    return {"t_learner": t_learner, "s_learner": s_learner, "x_learner": x_learner, "dr_learner": dr_learner}


def estimate_ates(data: pd.DataFrame) -> dict[str, float]:
    treatment, outcome, x = data["treatment_bonus"].to_numpy(float), data["future_ggr_eur"].to_numpy(float), design_matrix(data)
    naive = outcome[treatment == 1].mean() - outcome[treatment == 0].mean()
    regression = _ols(np.column_stack([x, treatment]), outcome)[-1]
    propensity = _logistic(x, treatment)
    ipw = np.mean(treatment * outcome / propensity - (1 - treatment) * outcome / (1 - propensity))
    treated, control = np.where(treatment == 1)[0], np.where(treatment == 0)[0]
    matched = np.mean([outcome[index] - outcome[control[np.argmin(np.abs(propensity[control] - propensity[index]))]] for index in treated])
    mu1 = x @ _ols(x[treatment == 1], outcome[treatment == 1])
    mu0 = x @ _ols(x[treatment == 0], outcome[treatment == 0])
    doubly_robust = np.mean(mu1 - mu0 + treatment / propensity * (outcome - mu1) - (1 - treatment) / (1 - propensity) * (outcome - mu0))
    estimates = {"naive": float(naive), "regression_adjustment": float(regression), "propensity_score_matching": float(matched), "ipw": float(ipw), "doubly_robust": float(doubly_robust), "dml": dml_ate(x, treatment, outcome)}
    estimates.update({name: float(cate.mean()) for name, cate in meta_learners(x, treatment, outcome).items()})
    return estimates


def evaluate(data: pd.DataFrame, level: str) -> pd.DataFrame:
    truth = float(data["true_ate_eur"].iloc[0])
    true_cate = data["true_individual_effect_eur"].to_numpy(float)
    x, treatment, outcome = design_matrix(data), data["treatment_bonus"].to_numpy(float), data["future_ggr_eur"].to_numpy(float)
    cate_estimates = meta_learners(x, treatment, outcome)
    estimates = estimate_ates(data)
    rows = []
    for method, estimate in estimates.items():
        cate = cate_estimates.get(method, np.full(len(data), estimate))
        rows.append({"level": level, "method": method, "estimate_ate_eur": estimate, "true_ate_eur": truth, "bias_eur": estimate - truth, "absolute_error_eur": abs(estimate - truth), "cate_rmse_eur": float(np.sqrt(np.mean((cate - true_cate) ** 2)))})
    return pd.DataFrame(rows)
