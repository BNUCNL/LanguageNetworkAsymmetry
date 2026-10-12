"""Estimate the age patterns and association of the two hemispheric asymmetries."""

import stats
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def run(data, output_dir, definition="14v9"):
    output = stats.output_directory(output_dir)
    specifications = {
        "connectivity_age": dict(
            outcome=stats.CONNECTIVITY, predictor=None, motions=[stats.REST_MOTION]
        ),
        "activation_age": dict(
            outcome=stats.ACTIVATION, predictor=None, motions=[stats.TASK_MOTION]
        ),
        "connectivity_activation": dict(
            outcome=stats.ACTIVATION,
            predictor=stats.CONNECTIVITY,
            motions=[stats.REST_MOTION, stats.TASK_MOTION],
        ),
    }
    fits, rows, full, curves, scatter = stats.estimate_models(data, specifications)
    effect_sizes = []
    for label, column in [
        ("connectivity", stats.CONNECTIVITY),
        ("activation", stats.ACTIVATION),
    ]:
        values = data[column].to_numpy(float)
        name = f"{label}_mean"
        summary = stats.mean_test(values)
        rows.append(
            dict(
                model=name, term="mean", n=len(data), correction_family=None, **summary
            )
        )
        effect_sizes.append(
            dict(
                measure=label,
                n=len(data),
                mean=summary["estimate"],
                sd=summary["sd"],
                cohens_d=summary["cohens_d"],
            )
        )
    stats.apply_bonferroni(rows).to_csv(output / "coefficients.csv", index=False)
    full.to_csv(output / "full_coefficients.csv", index=False)
    if effect_sizes:
        pd.DataFrame(effect_sizes).to_csv(output / "effect_sizes.csv", index=False)
    curves.to_csv(output / "age_curves.csv", index=False)
    selected = scatter.model.eq("connectivity_age")
    scatter.loc[selected, "visible"] = scatter.loc[selected, "observed"].between(-3, 5)
    scatter.to_csv(output / "age_scatter.csv", index=False)
    x, y = stats.partial_scatter(fits["connectivity_activation"])
    pd.DataFrame(
        dict(sub_id=data.sub_id, predictor_residual=x, outcome_residual=y, visible=True)
    ).to_csv(output / "association_scatter.csv", index=False)
    grid = np.linspace(x.min(), x.max(), 241)
    slope = fits["connectivity_activation"]["coefficients"]["predictor"]
    fit = fits["connectivity_activation"]
    index = list(fit["coefficients"]).index("predictor")
    width = (
        student_t.ppf(0.975, fit["df"])
        * np.sqrt(fit["covariance"][index, index])
        * np.abs(grid)
    )
    pd.DataFrame(
        dict(
            x=grid,
            estimate=slope * grid,
            ci95_lower=slope * grid - width,
            ci95_upper=slope * grid + width,
        )
    ).to_csv(output / "association_curve.csv", index=False)
    stats.write_metadata(
        output, data, None, None, dict(network_definition=definition, **specifications)
    )
    return output


def main():
    args = stats.analysis_arguments(__doc__)
    config = stats.read_config(args.config)
    data = stats.load_sample(stats.input_path(config, "subject_metrics"))
    if len(data) != 1167:
        raise ValueError("The manuscript analysis requires 1167 participants")
    run(data, stats.output_path(config, args.output_dir))


if __name__ == "__main__":
    main()
