"""Shared statistical functions for the language-network analyses."""

from __future__ import annotations
import scipy
import argparse
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

AGE = "Age_continuous"
SEX = "sex_num"
READING = "三分钟快速阅读_总正确率（%）"
CONNECTIVITY = "gradient_mean_asymmetry"
ACTIVATION = "rs_minus_cb"
REST_MOTION = "rest_motion"
TASK_MOTION = "task_motion"
AGE_TERMS = ("age_linear", "age_quadratic")
ASSOCIATION_TERMS = (
    "predictor",
    "age_linear",
    "age_quadratic",
    "predictor_x_age_linear",
    "predictor_x_age_quadratic",
)


def read_csv(path):
    return pd.read_csv(path, dtype={"sub_id": str}, float_precision="round_trip")


def load_sample(path, required=None):
    data = read_csv(path)
    columns = required or [
        AGE,
        SEX,
        "site_code",
        CONNECTIVITY,
        ACTIVATION,
        READING,
        REST_MOTION,
        TASK_MOTION,
    ]
    if data.sub_id.isna().any() or not data.sub_id.is_unique:
        raise ValueError("Participant IDs must be present and unique")
    if not np.isfinite(data[columns].to_numpy(float)).all():
        raise ValueError("Missing or nonfinite model input")
    if not data[SEX].isin([0, 1]).all():
        raise ValueError("Sex must use the study coding: female=0, male=1")
    if not data[AGE].between(6.0, 16.5).all():
        raise ValueError("Age outside the study eligibility range")
    return data.reset_index(drop=True)


def output_directory(path):
    path = Path(path).resolve()
    if path == Path(__file__).resolve().parent:
        raise ValueError("Choose an output subdirectory, not the code directory itself")
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def analysis_arguments(description, output_dir, bootstrap=False):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--output-dir", type=Path, default=output_dir)
    if bootstrap:
        parser.add_argument("--bootstraps", type=int, default=10000)
        parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def zscore(values):
    values = np.asarray(values, dtype=float)
    sd = values.std(ddof=1)
    if not np.isfinite(values).all() or sd <= 0:
        raise ValueError("Standardization requires finite, nonconstant data")
    return (values - values.mean()) / sd


def age_basis(age):
    linear = zscore(age)
    design = np.column_stack([np.ones(len(linear)), linear])
    projection = np.linalg.lstsq(design, linear**2, rcond=None)[0]
    residual = linear**2 - design @ projection
    sd = residual.std(ddof=1)
    if sd <= 0:
        raise ValueError("Quadratic age is not estimable")
    parameters = dict(
        mean=float(np.mean(age)),
        sd=float(np.std(age, ddof=1)),
        intercept=float(projection[0]),
        slope=float(projection[1]),
        residual_sd=float(sd),
    )
    return (linear, residual / sd, parameters)


def evaluate_age(age, parameters):
    linear = (np.asarray(age) - parameters["mean"]) / parameters["sd"]
    return (linear, evaluate_quadratic(linear, parameters))


def evaluate_quadratic(linear, parameters):
    return (
        np.asarray(linear) ** 2 - parameters["intercept"] - parameters["slope"] * linear
    ) / parameters["residual_sd"]


def controls(data, linear, quadratic, motions, categories):
    values = {"sex": data[SEX].to_numpy(float)}
    sites = data.site_code.to_numpy(int)
    for category in categories[1:]:
        values[f"site_{category}"] = (sites == category).astype(float)
    age_design = np.column_stack([np.ones(len(data)), linear, quadratic])
    for name in motions:
        motion = data[name].to_numpy(float)
        residual = (
            motion - age_design @ np.linalg.lstsq(age_design, motion, rcond=None)[0]
        )
        values[name] = zscore(residual)
    return values


def ols(y, predictors):
    names = ["intercept", *predictors]
    design = np.column_stack([np.ones(len(y)), *predictors.values()])
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ coefficients
    total = y - np.mean(y)
    return (dict(zip(names, coefficients)), 1 - residual @ residual / (total @ total))


def regression(data, outcome, predictor, motions, categories):
    linear, quadratic, parameters = age_basis(data[AGE])
    terms = {}
    if predictor is not None:
        x = zscore(data[predictor])
        terms.update(
            predictor=x,
            predictor_x_age_linear=x * linear,
            predictor_x_age_quadratic=x * quadratic,
        )
    terms.update(age_linear=linear, age_quadratic=quadratic)
    terms.update(controls(data, linear, quadratic, motions, categories))
    y = zscore(data[outcome])
    coefficients, r_squared = ols(y, terms)
    return dict(
        coefficients=coefficients,
        r_squared=r_squared,
        terms=terms,
        age_parameters=parameters,
        y=y,
        outcome_mean=float(data[outcome].mean()),
        outcome_sd=float(data[outcome].std(ddof=1)),
    )


def bootstrap_indices(data, count=10000, seed=42):
    if count < 2:
        raise ValueError("At least two bootstrap resamples are required")
    rng = np.random.default_rng(seed)
    sites = data.site_code.to_numpy(int)
    blocks = []
    for site in sorted(np.unique(sites)):
        positions = np.flatnonzero(sites == site)
        blocks.append(
            positions[rng.integers(0, len(positions), size=(count, len(positions)))]
        )
    return np.concatenate(blocks, axis=1)


def summary(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite bootstrap estimates")
    lower = (np.count_nonzero(values <= 0) + 1) / (len(values) + 1)
    upper = (np.count_nonzero(values >= 0) + 1) / (len(values) + 1)
    limits = np.quantile(values, [0.025, 0.975])
    return dict(
        bootstrap_mean=float(values.mean()),
        bootstrap_se=float(values.std(ddof=1)),
        ci95_lower=float(limits[0]),
        ci95_upper=float(limits[1]),
        p_raw=float(min(1, 2 * min(lower, upper))),
    )


def apply_bonferroni(rows):
    table = pd.DataFrame(rows)
    if not table.p_raw.between(0, 1).all():
        raise ValueError("Invalid raw p values")
    table["p_corrected"] = table.p_raw
    table["p_bonferroni"] = np.nan
    table["family_n"] = 0
    table["correction"] = "none"
    for _, positions in table.groupby("correction_family").groups.items():
        corrected = np.minimum(1.0, table.loc[positions, "p_raw"] * len(positions))
        table.loc[positions, ["p_corrected", "p_bonferroni"]] = np.column_stack(
            (corrected, corrected)
        )
        table.loc[positions, "family_n"] = len(positions)
        table.loc[positions, "correction"] = "bonferroni"
    return table


def parametric_inference(fit):
    design = np.column_stack([np.ones(len(fit["y"])), *fit["terms"].values()])
    beta = np.array(list(fit["coefficients"].values()))
    rank = np.linalg.matrix_rank(design)
    if rank != design.shape[1]:
        raise ValueError("Rank-deficient design")
    df = len(design) - rank
    residual = fit["y"] - design @ beta
    inverse = np.linalg.pinv(design)
    covariance = residual @ residual / df * (inverse @ inverse.T)
    fit.update(covariance=covariance, df=df, design=design)
    se = np.sqrt(np.diag(covariance))
    critical = student_t.ppf(0.975, df)
    return {
        name: dict(
            se=float(se[i]),
            t=float(beta[i] / se[i]),
            df=int(df),
            p_raw=float(2 * student_t.sf(abs(beta[i] / se[i]), df)),
            ci95_lower=float(beta[i] - critical * se[i]),
            ci95_upper=float(beta[i] + critical * se[i]),
        )
        for i, name in enumerate(fit["coefficients"])
    }


def mean_test(values):
    values = np.asarray(values, float)
    n = len(values)
    mean, sd = (values.mean(), values.std(ddof=1))
    if n < 2 or not np.isfinite(values).all() or sd <= 0:
        raise ValueError("Mean test requires finite, nonconstant observations")
    se = sd / np.sqrt(n)
    statistic, df = (mean / se, n - 1)
    critical = student_t.ppf(0.975, df)
    return dict(
        estimate=float(mean),
        sd=float(sd),
        se=float(se),
        t=float(statistic),
        df=df,
        p_raw=float(student_t.sf(statistic, df)),
        alternative="greater",
        ci95_lower=float(mean - critical * se),
        ci95_upper=float(mean + critical * se),
        cohens_d=float(mean / sd),
    )


def age_curve(fit, ages):
    linear, quadratic = evaluate_age(ages, fit["age_parameters"])
    beta = fit["coefficients"]
    baseline = beta["intercept"] + sum(
        (beta[k] * v.mean() for k, v in fit["terms"].items() if k not in AGE_TERMS)
    )
    return fit["outcome_mean"] + fit["outcome_sd"] * (
        baseline + beta["age_linear"] * linear + beta["age_quadratic"] * quadratic
    )


def age_scatter(fit):
    beta = fit["coefficients"]
    nuisance = sum(
        (
            beta[k] * (v - v.mean())
            for k, v in fit["terms"].items()
            if k not in AGE_TERMS
        )
    )
    return fit["outcome_mean"] + fit["outcome_sd"] * (fit["y"] - nuisance)


def partial_scatter(fit):
    x = fit["terms"]["predictor"]
    nuisance = np.column_stack(
        [np.ones(len(x)), *[v for k, v in fit["terms"].items() if k != "predictor"]]
    )
    xy = np.column_stack([x, fit["y"]])
    residuals = xy - nuisance @ np.linalg.lstsq(nuisance, xy, rcond=None)[0]
    return (residuals[:, 0], residuals[:, 1])


def estimate_models(data, specifications):
    categories = sorted(data.site_code.astype(int).unique())
    fits = {
        name: regression(data, **spec, categories=categories)
        for name, spec in specifications.items()
    }
    terms = {
        name: ASSOCIATION_TERMS if spec["predictor"] else AGE_TERMS
        for name, spec in specifications.items()
    }
    ages = np.linspace(float(data[AGE].min()), float(data[AGE].max()), 241)
    rows, full_rows = ([], [])
    for name, fit in fits.items():
        inference = parametric_inference(fit)
        for term, value in fit["coefficients"].items():
            row = dict(
                model=name,
                term=term,
                n=len(data),
                estimate=float(value),
                model_r_squared=float(fit["r_squared"]),
                alternative="two-sided",
                correction_family=name if term in terms[name] else None,
                **inference[term],
            )
            full_rows.append(row)
            if term in terms[name]:
                rows.append(row)
    curves, scatter = ([], [])
    for name, fit in fits.items():
        if specifications[name]["predictor"] is not None:
            continue
        linear, quadratic = evaluate_age(ages, fit["age_parameters"])
        design = np.column_stack(
            [
                np.ones(len(ages)),
                *[
                    (
                        linear
                        if term == "age_linear"
                        else (
                            quadratic
                            if term == "age_quadratic"
                            else np.full(len(ages), values.mean())
                        )
                    )
                    for term, values in fit["terms"].items()
                ],
            ]
        )
        se = (
            np.sqrt(np.einsum("ij,jk,ik->i", design, fit["covariance"], design))
            * fit["outcome_sd"]
        )
        curve = age_curve(fit, ages)
        width = student_t.ppf(0.975, fit["df"]) * se
        curves.append(
            pd.DataFrame(
                dict(
                    model=name,
                    age_years=ages,
                    estimate=curve,
                    ci95_lower=curve - width,
                    ci95_upper=curve + width,
                )
            )
        )
        scatter.append(
            pd.DataFrame(
                dict(
                    model=name,
                    sub_id=data.sub_id,
                    age_years=data[AGE],
                    observed=data[specifications[name]["outcome"]],
                    adjusted=age_scatter(fits[name]),
                    visible=True,
                )
            )
        )
    return (
        fits,
        rows,
        apply_bonferroni(full_rows),
        pd.concat(curves, ignore_index=True),
        pd.concat(scatter, ignore_index=True),
    )


def write_metadata(output, data, indices, seed, specifications):
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    versions["scipy"] = scipy.__version__
    count = 0 if indices is None else len(indices)
    metadata = dict(
        n=len(data),
        bootstraps=count,
        seed=seed,
        formal_settings=len(data) == 1167
        and (count == 0 or (count == 10000 and seed == 42)),
        bootstrap_indices_sha256=(
            None if indices is None else hashlib.sha256(indices.tobytes()).hexdigest()
        ),
        sample_order_sha256=hashlib.sha256("\n".join(data.sub_id).encode()).hexdigest(),
        sample_values_sha256=hashlib.sha256(
            data.to_csv(index=False).encode()
        ).hexdigest(),
        age_range=[float(data[AGE].min()), float(data[AGE].max())],
        models=specifications,
        software=versions,
        ci=(
            "95% percentile participant bootstrap, stratified by acquisition site"
            if count
            else "95% Student t, ordinary nonrobust OLS; two-sided mean intervals"
        ),
        inference="site-stratified bootstrap" if count else "Student t",
        multiplicity="Bonferroni within each regression; uncorrected mean and derived indirect tests",
        standardization="Sample SD (ddof=1); recomputed within each resample",
        age_basis="Standardized age and zero-mean orthogonal quadratic age",
        coefficient_scale="Standardized regression coefficients; means and Cohen's d are identified separately",
        motion_processing="FD residualized on both age terms and standardized within each model resample",
        script_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).resolve().parent.glob("*.py")
        },
    )
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
