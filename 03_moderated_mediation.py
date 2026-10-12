"""Age-moderated indirect associations with reading (Results 3.3)."""

import stats
from pathlib import Path
import numpy as np
import pandas as pd

PATHS = ("a", "b", "c_prime", "c")
ANCHORS = {"low": -1.0, "mean": 0.0, "high": 1.0}
GROUPS = ((6, 9, "6-8"), (9, 12, "9-11"), (12, 16, "12-15"))


def fit_model(data, categories):
    linear, quadratic, parameters = stats.age_basis(data[stats.AGE])
    x, m, y = (
        stats.zscore(data[name])
        for name in (stats.CONNECTIVITY, stats.ACTIVATION, stats.READING)
    )
    controls = dict(
        age_linear=linear,
        age_quadratic=quadratic,
        **stats.controls(
            data, linear, quadratic, [stats.REST_MOTION, stats.TASK_MOTION], categories
        ),
    )
    x_terms = dict(x=x, x_linear=x * linear, x_quadratic=x * quadratic)
    m_terms = dict(m=m, m_linear=m * linear, m_quadratic=m * quadratic)
    mediator, _ = stats.ols(m, {**x_terms, **controls})
    outcome, _ = stats.ols(y, {**x_terms, **m_terms, **controls})
    total, _ = stats.ols(y, {**x_terms, **controls})
    coefficients = {
        "a": (mediator, "x"),
        "b": (outcome, "m"),
        "c_prime": (outcome, "x"),
        "c": (total, "x"),
    }
    point, conditional = ({}, {})
    for name, values in (
        ("mediator", mediator),
        ("outcome", outcome),
        ("total", total),
    ):
        for age_term in stats.AGE_TERMS:
            point[f"{name}_{age_term}"] = float(values[age_term])
    for path, (values, name) in coefficients.items():
        slope = (
            values[name]
            + values[f"{name}_linear"] * linear
            + values[f"{name}_quadratic"] * quadratic
        )
        conditional[path] = slope
        point[f"{path}_average"] = float(slope.mean())
        point[f"{path}_age_linear"] = float(values[f"{name}_linear"])
        point[f"{path}_age_quadratic"] = float(values[f"{name}_quadratic"])
    product = conditional["a"] * conditional["b"]
    point["age_distribution_average_indirect"] = float(product.mean())
    for label, value in ANCHORS.items():
        q = stats.evaluate_quadratic(value, parameters)
        a = mediator["x"] + mediator["x_linear"] * value + mediator["x_quadratic"] * q
        b = outcome["m"] + outcome["m_linear"] * value + outcome["m_quadratic"] * q
        point[f"indirect_{label}"] = float(a * b)
    point["high_minus_low_indirect"] = point["indirect_high"] - point["indirect_low"]
    ages = data[stats.AGE].to_numpy(float)
    groups, counts = ({}, {})
    for lower, upper, label in GROUPS:
        selected = (ages >= lower) & (ages < upper)
        if not selected.any():
            raise ValueError(f"No participants in age group {label}")
        groups[label] = float(product[selected].mean())
        counts[label] = int(selected.sum())
    return (point, groups, counts)


def run(data, output_dir, count=10000, seed=42):
    output = stats.output_directory(output_dir)
    categories = sorted(data.site_code.astype(int).unique())
    indices = stats.bootstrap_indices(data, count, seed)
    point, groups, counts = fit_model(data, categories)
    draws = {term: np.empty(count) for term in point}
    group_draws = {label: np.empty(count) for label in groups}
    group_counts = {label: np.empty(count, dtype=int) for label in groups}
    for draw, selected in enumerate(indices):
        estimates, age_groups, ns = fit_model(
            data.iloc[selected].reset_index(drop=True), categories
        )
        for term, estimate in estimates.items():
            draws[term][draw] = estimate
        for label in groups:
            group_draws[label][draw], group_counts[label][draw] = (
                age_groups[label],
                ns[label],
            )
        if (draw + 1) % 1000 == 0:
            print(f"Bootstrap {draw + 1}/{count}", flush=True)
    rows = []
    for term, estimate in point.items():
        if term.startswith("indirect_"):
            continue
        if term.startswith(("mediator_", "outcome_", "total_")):
            family = term.split("_", 1)[0]
        elif term.startswith("a_"):
            family = "mediator"
        elif term.startswith(("b_", "c_prime_")):
            family = "outcome"
        elif term.startswith("c_"):
            family = "total"
        else:
            family = None
        rows.append(
            dict(
                model="mediation",
                term=term,
                n=len(data),
                estimate=estimate,
                correction_family=family,
                **stats.summary(draws[term]),
            )
        )
    stats.apply_bonferroni(rows).to_csv(output / "coefficients.csv", index=False)
    pd.DataFrame({f"mediation__{k}": v for k, v in draws.items()}).to_csv(
        output / "bootstrap_draws.csv", index=False
    )
    conditional_rows = []
    for label, value in ANCHORS.items():
        term = f"indirect_{label}"
        summary = stats.summary(draws[term])
        summary.pop("p_raw")
        conditional_rows.append(
            dict(
                age_label=label,
                age_linear=value,
                age_years=float(
                    data[stats.AGE].mean() + value * data[stats.AGE].std(ddof=1)
                ),
                n=len(data),
                estimate=point[term],
                **summary,
            )
        )
    pd.DataFrame(conditional_rows).to_csv(
        output / "conditional_indirect.csv", index=False
    )
    group_rows = []
    for lower, upper, label in GROUPS:
        summary = stats.summary(group_draws[label])
        summary.pop("p_raw")
        group_rows.append(
            dict(
                age_group=label,
                bin_lower=lower,
                bin_upper_exclusive=upper,
                n=counts[label],
                model_n=len(data),
                estimate=groups[label],
                **summary,
            )
        )
    pd.DataFrame(group_rows).to_csv(output / "age_group_indirect.csv", index=False)
    pd.DataFrame(group_draws).to_csv(output / "age_group_draws.csv", index=False)
    pd.DataFrame(group_counts).to_csv(output / "age_group_counts.csv", index=False)
    stats.write_metadata(
        output,
        data,
        indices,
        seed,
        {
            "mediator": "M ~ X * (L + Q) + sex + site + rest FD + task FD",
            "outcome": "Y ~ (X + M) * (L + Q) + sex + site + rest FD + task FD",
            "total": "Y ~ X * (L + Q) + sex + site + rest FD + task FD",
            "X": stats.CONNECTIVITY,
            "M": stats.ACTIVATION,
            "Y": stats.READING,
            "average_indirect": "Mean of a(L,Q)*b(L,Q) over observed participant ages",
            "age_contrast": "Conditional indirect association at +1 SD minus -1 SD age",
            "group_display": "Mean conditional product within each observed age group; no subgroup fits",
        },
    )
    return output


def main():
    args = stats.analysis_arguments(__doc__, bootstrap=True)
    config = stats.read_config(args.config)
    data = stats.load_sample(stats.input_path(config, "subject_metrics"))
    if len(data) != 1167:
        raise ValueError("The manuscript analysis requires 1167 participants")
    run(data, stats.output_path(config, args.output_dir), args.bootstraps, args.seed)


if __name__ == "__main__":
    main()
