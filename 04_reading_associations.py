"""Estimate reading age patterns and the two separate brain-reading associations."""

import stats
from pathlib import Path
import numpy as np

METRICS_FILE = Path("results/metrics/subject_metrics.csv")
OUTPUT_DIR = Path("results/reading")


def run(data, output_dir):
    output = stats.output_directory(output_dir)
    specifications = {
        "reading_age": dict(outcome=stats.READING, predictor=None, motions=[]),
        "connectivity_reading": dict(
            outcome=stats.READING,
            predictor=stats.CONNECTIVITY,
            motions=[stats.REST_MOTION],
        ),
        "activation_reading": dict(
            outcome=stats.READING,
            predictor=stats.ACTIVATION,
            motions=[stats.TASK_MOTION],
        ),
    }
    fits, rows, full, curves, scatter = stats.estimate_models(data, specifications)
    stats.apply_bonferroni(rows).to_csv(output / "coefficients.csv", index=False)
    full.to_csv(output / "full_coefficients.csv", index=False)
    curves.to_csv(output / "age_curves.csv", index=False)
    fit = fits["reading_age"]
    design = np.column_stack([np.ones(len(data)), *fit["terms"].values()])
    beta = np.array(list(fit["coefficients"].values()))
    residual = fit["y"] - design @ beta
    leverage = np.sum(design * np.linalg.pinv(design).T, axis=1)
    variance = (residual @ residual - residual**2 / (1 - leverage)) / (
        len(data) - design.shape[1] - 1
    )
    studentized = residual / np.sqrt(variance * (1 - leverage))
    scatter["visible"] = (np.abs(studentized) <= 3) & (scatter.observed < 100)
    scatter.to_csv(output / "age_scatter.csv", index=False)
    stats.write_metadata(output, data, None, None, specifications)
    return output


def main():
    args = stats.analysis_arguments(__doc__, OUTPUT_DIR, bootstrap=False)
    data = stats.load_sample(METRICS_FILE)
    if len(data) != 1167:
        raise ValueError("The manuscript analysis requires 1167 participants")
    run(data, args.output_dir)


if __name__ == "__main__":
    main()
