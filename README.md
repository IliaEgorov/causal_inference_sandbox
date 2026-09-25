# Causal inference sandbox

This project turns a real-looking iGaming transaction ledger into a controlled causal-learning environment. The source files in `data/` are read-only inputs; all generated datasets and reports are written to `outputs/`.

## Important data limitation

The supplied ledger has **156,672 transaction events and 417 sessions, but no user identifier**. It appears to describe one account over time, not a panel of independent real users. The preparation step therefore creates one *synthetic analytical unit per real session*. Its pre-session activity becomes covariates and its later-in-session GGR becomes the natural outcome component. `country` and `acquisition_channel` are explicitly labelled synthetic strata; they are deterministic balancing features, not claims about the source data.

This is appropriate for comparing estimators against known ground truth, but it must not be used to infer a causal bonus effect in the source account.

## Data-generating process

Treatment means an assigned bonus. The observed outcome is `natural_future_ggr_eur + true_individual_effect_eur * treatment`. No estimator receives the true effect or assignment probability.

| Level | Assignment | Individual effect |
| --- | --- | --- |
| `01_randomized` | 50/50 random | €10 for every unit |
| `02_observable_confounding` | Depends on historical value and synthetic strata | €8 for every unit |
| `03_nonlinear_confounding` | Nonlinear value/activity interactions | €8 for every unit |
| `04_heterogeneous_effect` | Nonlinear confounding | VIP €30, regular €10, low-value -€2 |

The report compares naive, regression adjustment, propensity-score matching, IPW, doubly robust, and cross-fitted DML estimates with the known ATE. It also runs T-, S-, X-, and DR-learners. Every row includes `cate_rmse_eur`, the error of the estimated individual effect against hidden ground truth.

## Run

From the repository root:

```bash
./events/.venv/bin/python causal_inference_sandbox/run.py
```

Optional: execute a single level.

```bash
./events/.venv/bin/python causal_inference_sandbox/run.py --level 03_nonlinear_confounding
```

Outputs:

- `outputs/session_units.csv` — derived, session-level feature table;
- `outputs/<level>_dataset.csv` — simulated campaign data; and
- `outputs/evaluation.csv` — estimate, true ATE, bias, and absolute error by method.

## Notebook

Open [notebooks/01_causal_sandbox.ipynb](notebooks/01_causal_sandbox.ipynb) in Jupyter. It prepares the units, runs all difficulty levels, ranks estimators by ATE error, and inspects heterogeneous effects. It uses only `numpy` and `pandas`, already available in `events/.venv`.

## Weekly MMM

[mmm.py](mmm.py) fits a weekly FTD marketing mix model from a CSV shaped as `week`, `FTDs`, `Channel_Spend_*`, and optional `Confounder_*`. It automatically detects channels and confounders, applies geometric adstock and Hill saturation to channel spend, controls for trend and annual seasonality, and uses chronological holdout validation to select the adstock parameter and ridge penalty. Media coefficients are constrained to be non-negative.

```bash
./events/.venv/bin/python causal_inference_sandbox/mmm.py path/to/weekly_data.csv \
  --output-dir causal_inference_sandbox/outputs/mmm
```

It writes model parameters, weekly channel contributions, and per-spend ROI to the output directory. MMM measures associations under its modelling assumptions; confounders should include material demand, product, pricing, and measurement changes before interpreting channel contributions causally.

### Start with the simple version

[mmm_simple.py](mmm_simple.py) is the learning version: a plain linear regression on raw channel spend and optional confounders. It has no carry-over, diminishing returns, seasonality, regularisation, or non-negative constraints.

```bash
./events/.venv/bin/python causal_inference_sandbox/mmm_simple.py path/to/weekly_data.csv \
  --output-dir causal_inference_sandbox/outputs/mmm_simple
```

Try it immediately with [data/weekly_mmm_example.csv](data/weekly_mmm_example.csv).

Suggested upgrade sequence: add a trend and seasonal controls, add one adstock parameter, add saturation, then move to holdout validation and regularisation in `mmm.py`.
