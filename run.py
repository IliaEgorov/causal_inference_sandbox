"""Run the controlled causal-inference laboratory from raw event data."""

from __future__ import annotations

import argparse
from pathlib import Path

from sandbox import ROOT, evaluate, prepare_session_units, simulate_campaign


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", choices=["01_randomized", "02_observable_confounding", "03_nonlinear_confounding", "04_heterogeneous_effect"])
    parser.add_argument("--seed", type=int, default=27)
    args = parser.parse_args()
    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    units = prepare_session_units()
    units.to_csv(output / "session_units.csv", index=False)
    levels = [args.level] if args.level else ["01_randomized", "02_observable_confounding", "03_nonlinear_confounding", "04_heterogeneous_effect"]
    reports = []
    for level in levels:
        dataset = simulate_campaign(units, level, args.seed)
        dataset.to_csv(output / f"{level}_dataset.csv", index=False)
        reports.append(evaluate(dataset, level))
    report = __import__("pandas").concat(reports, ignore_index=True)
    report.to_csv(output / "evaluation.csv", index=False)
    print(report.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
