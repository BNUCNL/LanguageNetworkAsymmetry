"""Age patterns and associations of hemispheric asymmetries (Results 3.1)."""

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


def curve_correlations(main_dir, homotopic_dir):
    main = pd.read_csv(Path(main_dir) / "age_curves.csv", float_precision="round_trip")
    paired = pd.read_csv(
        Path(homotopic_dir) / "age_curves.csv", float_precision="round_trip"
    )
    rows = []
    for measure in ("connectivity", "activation"):
        a = main[main.model.eq(measure + "_age")].sort_values("age_years")
        b = paired[paired.model.eq(measure + "_age")].sort_values("age_years")
        if len(a) != 241 or not np.array_equal(a.age_years, b.age_years):
            raise ValueError(
                "Developmental curves must share the 241 observed-range age points"
            )
        if not np.isfinite(a.estimate).all() or not np.isfinite(b.estimate).all():
            raise ValueError("Nonfinite fitted curves")
        rows.append(
            dict(
                measure=measure,
                pearson_r=float(np.corrcoef(a.estimate, b.estimate)[0, 1]),
                curve_points=len(a),
                age_min=a.age_years.min(),
                age_max=a.age_years.max(),
                inference="descriptive only; no p values or confidence intervals",
            )
        )
    target = Path(homotopic_dir) / "curve_correlations.csv"
    if target.exists():
        raise FileExistsError(target)
    pd.DataFrame(rows).to_csv(target, index=False)


def gradient_spectrum(data, path, output):
    eigenvalues = stats.read_csv(path)
    if not eigenvalues.sub_id.is_unique or set(eigenvalues.sub_id) != set(data.sub_id):
        raise ValueError("Gradient eigenvalues must match the analysis sample")
    eigenvalues = eigenvalues.set_index("sub_id").loc[data.sub_id]
    values = eigenvalues[[f"lambda_{i}" for i in range(1, 11)]].to_numpy(float)
    if (
        not np.isfinite(values).all()
        or (values < 0).any()
        or (values.sum(axis=1) <= 0).any()
    ):
        raise ValueError("Invalid diffusion-map eigenvalues")
    shares = values / values.sum(axis=1, keepdims=True) * 100
    output = stats.output_directory(output)
    pd.DataFrame(
        dict(
            sub_id=np.repeat(data.sub_id.to_numpy(), 10),
            gradient=np.tile(np.arange(1, 11), len(data)),
            percent=shares.ravel(),
        )
    ).to_csv(output / "participant_percentages.csv", index=False)
    pd.DataFrame(
        dict(
            gradient=np.arange(1, 11),
            n=len(data),
            mean_percent=shares.mean(axis=0),
            sd_percent=shares.std(axis=0, ddof=1),
            median_percent=np.median(shares, axis=0),
        )
    ).to_csv(output / "summary.csv", index=False)


def run_homologous(config, data, main_output):
    path = stats.input_path(config, "homotopic_metrics")
    paired = stats.load_sample(path)
    if not paired.sub_id.equals(data.sub_id):
        raise ValueError("Homologous metrics must preserve participant order")
    columns = [
        stats.AGE,
        stats.SEX,
        "site_code",
        stats.REST_MOTION,
        stats.TASK_MOTION,
        stats.READING,
        "array_row",
        "task_run_n",
        "task_motion_run_n",
    ]
    for column in columns:
        if column not in data or column not in paired:
            raise ValueError(f"Missing sample field: {column}")
        if not np.array_equal(data[column].to_numpy(), paired[column].to_numpy()):
            raise ValueError(f"Homologous analysis changed {column}")
    pairs = stats.read_csv(path.parent / "homotopic_language_pairs.csv")
    if len(pairs) != 9 or not pairs.base_label.is_unique:
        raise ValueError("Expected nine homologous language-network pairs")
    if not np.array_equal(pairs.right_index_1based, pairs.left_index_1based + 180):
        raise ValueError("Homologous parcel indices do not match")
    output = Path(main_output).parent / "supplementary_9v9"
    run(paired, output, definition="9v9")
    curve_correlations(main_output, output)


def main():
    args = stats.analysis_arguments(__doc__)
    config = stats.read_config(args.config)
    data = stats.load_sample(stats.input_path(config, "subject_metrics"))
    if len(data) != 1167:
        raise ValueError("The manuscript analysis requires 1167 participants")
    output = run(data, stats.output_path(config, args.output_dir))
    if "homotopic_metrics" in config["inputs"]:
        run_homologous(config, data, output)
    if "diffusion_eigenvalues" in config["inputs"]:
        gradient_spectrum(
            data,
            stats.input_path(config, "diffusion_eigenvalues"),
            output.parent / "spectrum",
        )


if __name__ == "__main__":
    main()
