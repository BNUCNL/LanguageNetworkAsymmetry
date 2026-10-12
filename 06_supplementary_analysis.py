"""Run the homologous-parcel analysis or summarize the first ten diffusion eigenvalues."""

from __future__ import annotations
import hashlib
import json
import argparse
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pandas as pd


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


def load_module(filename):
    path = Path(__file__).resolve().with_name(filename)
    name = "release_" + path.stem
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


prep = load_module("01_data_preparation.py")


def summarize_spectrum(config, config_path, output_dir):
    paths = prep.input_paths(
        config, config_path, ["subject_metrics", "diffusion_eigenvalues"], []
    )
    sample = prep.read_table(paths["subject_metrics"])
    source = prep.read_table(paths["diffusion_eigenvalues"])
    prep.require(
        sample.sub_id.is_unique and source.sub_id.is_unique,
        "Duplicate spectrum/sample IDs",
    )
    prep.require(
        set(sample.sub_id) == set(source.sub_id),
        "Spectrum sample differs from analysis sample",
    )
    source = source.set_index("sub_id").loc[sample.sub_id]
    columns = [f"lambda_{i}" for i in range(1, 11)]
    values = source[columns].to_numpy(float)
    prep.require(
        np.isfinite(values).all()
        and (values >= 0).all()
        and (values.sum(axis=1) > 0).all(),
        "Invalid diffusion eigenvalues",
    )
    shares = values / values.sum(axis=1, keepdims=True) * 100
    output = prep.new_output(config_path, output_dir)
    long = pd.DataFrame(
        dict(
            sub_id=np.repeat(sample.sub_id.to_numpy(), 10),
            gradient=np.tile(np.arange(1, 11), len(sample)),
            percent=shares.ravel(),
        )
    )
    long.to_csv(output / "participant_percentages.csv", index=False)
    pd.DataFrame(
        dict(
            gradient=np.arange(1, 11),
            n=len(sample),
            mean_percent=shares.mean(axis=0),
            sd_percent=shares.std(axis=0, ddof=1),
            median_percent=np.median(shares, axis=0),
        )
    ).to_csv(output / "summary.csv", index=False)
    record = dict(
        n=len(sample),
        denominator="sum of the first ten BrainSpace diffusion eigenvalues",
        aligned_coordinate_variance=False,
        eigenvalues_squared=False,
        input_sha256={
            k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in paths.items()
        },
        sample_order_sha256=hashlib.sha256(
            "\n".join(sample.sub_id).encode()
        ).hexdigest(),
    )
    (output / "metadata.json").write_text(json.dumps(record, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--main-results-dir",
        type=Path,
        help="Optional main asymmetry output for descriptive curve correlations",
    )
    parser.add_argument(
        "--spectrum-only",
        action="store_true",
        help="Summarize supplied diffusion eigenvalues instead of the homologous-parcel models",
    )
    args = parser.parse_args()
    if args.spectrum_only:
        config, config_path = prep.load_config(args.config)
        summarize_spectrum(config, config_path, args.output_dir)
        return
    shared = load_module("stats.py")
    analysis = load_module("03_developmental_dissociation.py")
    config = shared.read_config(args.config)
    main_path = shared.input_path(config, "subject_metrics")
    paired_path = shared.input_path(config, "homotopic_metrics")
    if main_path.resolve() == paired_path.resolve():
        raise ValueError("Main and homotopic metrics must be separate files")
    original = shared.load_sample(main_path)
    paired = shared.load_sample(paired_path)
    pairs = shared.read_csv(paired_path.parent / "homotopic_language_pairs.csv")
    if len(pairs) != 9 or not pairs["base_label"].is_unique:
        raise ValueError("Expected nine unique bilateral language-network pairs")
    if not np.array_equal(
        pairs["right_index_1based"], pairs["left_index_1based"] + 180
    ):
        raise ValueError("Homotopic hemisphere indices differ")
    if not paired["sub_id"].equals(original["sub_id"]):
        raise ValueError(
            "Homotopic metrics must preserve the main sample and row order"
        )
    required = [
        shared.AGE,
        shared.SEX,
        "site_code",
        shared.REST_MOTION,
        shared.TASK_MOTION,
        shared.READING,
        "array_row",
        "task_run_n",
        "task_motion_run_n",
    ]
    for column in required:
        if column not in original or column not in paired:
            raise ValueError(f"Missing inherited sample field: {column}")
        if not np.array_equal(original[column].to_numpy(), paired[column].to_numpy()):
            raise ValueError(f"Homotopic analysis changed {column}")
    if len(paired) != 1167:
        raise ValueError(
            "The authorized homotopic analysis requires 1,167 participants"
        )
    output = shared.output_path(config, args.output_dir).resolve()
    if output in {main_path.resolve().parent, paired_path.resolve().parent}:
        raise ValueError("Use a separate output directory, not an input directory")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    analysis.run(paired, output, definition="9v9")
    if args.main_results_dir:
        curve_correlations(shared.output_path(config, args.main_results_dir), output)


if __name__ == "__main__":
    main()
