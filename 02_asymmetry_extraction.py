"""Extract L14-R9 and homologous L9-R9 asymmetries from aligned gradients and raw RS-CB beta."""

from __future__ import annotations
import argparse
import hashlib
import json
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

INPUTS = {
    "prepared_sample": "results/prepared/prepared_sample.csv",
    "retained_runs": "results/prepared/retained_runs.csv",
    "sample_flow": "results/prepared/sample_flow.csv",
    "gradients": "data/interim/brain_gradients_3981x360x3.npy",
    "brain_index": "data/processed/brain_subject_index.csv",
    "atlas": "external/atlases/ColeAnticevicNetPartition/CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR_LabelKey.txt",
    "left_handed": "data/processed/ccbd_left_handed_subjects.csv",
    "task_beta_run_level": "data/raw/task_fmri_mmp360/single_condition_beta_2026-08-31/task_cl_beta_mmp360_run_level.csv",
    "task_beta_subject_runmean": "data/raw/task_fmri_mmp360/single_condition_beta_2026-08-31/task_cl_beta_mmp360_subject_runmean.csv",
    "task_spm_audit": "data/raw/task_fmri_mmp360/single_condition_beta_2026-08-31/spm_design_audit.tsv",
}


def load_preparation():
    path = Path(__file__).resolve().with_name("01_data_preparation.py")
    spec = importlib.util.spec_from_file_location("ccbd_data_preparation", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load preparation utilities: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prep = load_preparation()


def language_members(path):
    atlas = prep.read_table(path).iloc[:360].copy()
    atlas["INDEX"] = prep.integers(atlas.INDEX, "atlas INDEX", 1)
    prep.require(
        np.array_equal(atlas.INDEX, np.arange(1, 361)),
        "Cortical atlas rows must be ordered 1..360",
    )
    prep.require(
        atlas.HEMISPHERE.tolist() == ["L"] * 180 + ["R"] * 180,
        "Cortical atlas hemisphere ordering must be L180 then R180",
    )
    prep.require(
        atlas.GLASSERLABELNAME.notna().all() and atlas.GLASSERLABELNAME.is_unique,
        "Missing or duplicate parcel label",
    )
    members = atlas.loc[atlas.NETWORK.eq("Language")].copy()
    prep.require(
        members.HEMISPHERE.value_counts().to_dict() == {"L": 14, "R": 9},
        "Expected original Cole-Anticevic Language L14/R9",
    )
    left = members.loc[members.HEMISPHERE.eq("L"), "INDEX"].to_numpy(int) - 1
    right = members.loc[members.HEMISPHERE.eq("R"), "INDEX"].to_numpy(int) - 1
    return (atlas, members, left, right)


def homotopic_pairs(atlas, members):
    frames = []
    for hemisphere in ("L", "R"):
        frame = members.loc[members.HEMISPHERE.eq(hemisphere)].copy()
        prep.require(
            frame.GLASSERLABELNAME.str.startswith(hemisphere + "_").all(),
            "Invalid hemisphere label prefix",
        )
        frame["base_label"] = frame.GLASSERLABELNAME.str[2:]
        prep.unique(frame, ["base_label"], "homotopic labels")
        prefix = "left" if hemisphere == "L" else "right"
        frames.append(
            frame[["base_label", "INDEX", "GLASSERLABELNAME"]].rename(
                columns={
                    "INDEX": prefix + "_index_1based",
                    "GLASSERLABELNAME": prefix + "_label",
                }
            )
        )
    pairs = frames[0].merge(
        frames[1], on="base_label", how="inner", validate="one_to_one"
    )
    pairs = pairs.sort_values("left_index_1based").reset_index(drop=True)
    prep.require(
        len(pairs) == 9
        and np.array_equal(pairs.right_index_1based, pairs.left_index_1based + 180),
        "Expected nine homologous pairs within bilateral Language",
    )
    return pairs


def extract(config, config_path, output_dir):
    paths = prep.input_paths(
        config,
        config_path,
        [
            "prepared_sample",
            "retained_runs",
            "gradients",
            "brain_index",
            "atlas",
            "left_handed",
            "task_beta_run_level",
            "task_spm_audit",
        ],
        ["task_beta_subject_runmean", "sample_flow"],
    )
    limits = prep.checks(config)
    options = config.get("extraction", {})
    include_9v9 = options.get("include_9v9", False)
    prep.require(
        isinstance(include_9v9, bool), "extraction.include_9v9 must be a JSON boolean"
    )
    sample = prep.ids(
        prep.read_table(paths["prepared_sample"]), "sub_id", "prepared sample"
    )
    prep.unique(sample, ["sub_id"], "prepared sample")
    prep.require(
        len(sample) == limits["expected_n"] and len(sample) > 0,
        "Prepared sample size mismatch",
    )
    prep.require(
        set(prep.SAMPLE_COLUMNS).issubset(sample),
        "Prepared sample columns are incomplete",
    )
    sample = sample[prep.SAMPLE_COLUMNS].copy()
    numeric_cols = [
        c
        for c in prep.SAMPLE_COLUMNS
        if c not in ("sub_id", prep.SEX_LABEL, prep.STAGE, "site", prep.REGION)
    ]
    sample[numeric_cols] = sample[numeric_cols].astype(float)
    prep.finite(sample, numeric_cols, "prepared sample")
    for col in (
        prep.GRADE,
        prep.SEX,
        "site_code",
        "array_row",
        "task_run_n",
        "task_motion_run_n",
        "task_motion_frame_n",
    ):
        sample[col] = prep.integers(sample[col], col)
    prep.require(
        sample[prep.AGE].between(6, 16.5).all()
        and sample[prep.GRADE].between(1, 9).all()
        and sample[prep.READING].between(0, 100).all()
        and set(sample[prep.SEX]).issubset({0, 1}),
        "Prepared sample eligibility/coding mismatch",
    )
    prep.require(
        (sample[["rest_motion", "task_motion"]].to_numpy() >= 0).all(), "Negative FD"
    )
    handedness = prep.left_handed_ids(paths["left_handed"], config)
    prep.require(
        not sample.sub_id.isin(handedness.sub_id).any(),
        "Left-handed participant in sample",
    )
    index = prep.brain_index(paths["brain_index"])
    rows = index.set_index("sub_id").loc[sample.sub_id, "array_row"].to_numpy(int)
    prep.require(
        np.array_equal(rows, sample.array_row),
        "Prepared subject-to-gradient row mapping differs",
    )
    gradients = np.load(paths["gradients"], mmap_mode="r", allow_pickle=False)
    expected_shape = tuple(limits.get("gradient_shape", [3981, 360, 3]))
    prep.require(
        gradients.shape == expected_shape
        and gradients.ndim == 3
        and (gradients.shape[0] == len(index))
        and (gradients.shape[1] == 360)
        and (gradients.shape[2] >= 1),
        "Aligned gradient shape/index mismatch",
    )
    gradient = np.asarray(gradients[rows, :, 0], dtype=float)
    prep.require(
        np.isfinite(gradient).all(), "Nonfinite principal gradient in prepared sample"
    )
    atlas, members, left, right = language_members(paths["atlas"])
    means, retained, beta_qc = prep.beta_tables(paths)
    retained = retained.loc[retained.sub_id.isin(sample.sub_id)].reset_index(drop=True)
    prepared_runs = prep.ids(
        prep.read_table(paths["retained_runs"]), "sub_id", "prepared retained runs"
    )
    prepared_runs["run"] = prep.integers(prepared_runs.run, "retained run", 1)
    prep.unique(prepared_runs, ["sub_id", "run"], "prepared retained runs")
    prepared_runs = (
        prepared_runs[["sub_id", "run"]]
        .sort_values(["sub_id", "run"])
        .reset_index(drop=True)
    )
    prep.require(
        prepared_runs.equals(retained),
        "Prepared retained runs differ from original beta export",
    )
    counts = retained.groupby("sub_id").size().loc[sample.sub_id].to_numpy()
    prep.require(
        np.array_equal(counts, sample.task_run_n)
        and np.array_equal(counts, sample.task_motion_run_n),
        "Task beta/FD run counts differ",
    )
    maps = {
        condition: means.loc[means.condition.eq(condition)]
        .set_index("sub_id")
        .loc[sample.sub_id, prep.REGIONS]
        .to_numpy(float)
        for condition in ("RS", "CB")
    }
    contrast = maps["RS"] - maps["CB"]

    def metrics(left_indices, right_indices):
        result = sample.copy()
        for name, values in [("gradient_mean", gradient), ("rs_minus_cb", contrast)]:
            left_mean = values[:, left_indices].mean(axis=1)
            right_mean = values[:, right_indices].mean(axis=1)
            result[name + ("_left" if name == "gradient_mean" else "__left")] = (
                left_mean
            )
            result[name + ("_right" if name == "gradient_mean" else "__right")] = (
                right_mean
            )
            result[
                "gradient_mean_asymmetry" if name == "gradient_mean" else "rs_minus_cb"
            ] = (left_mean - right_mean)
        return result[
            prep.SAMPLE_COLUMNS
            + [
                "gradient_mean_left",
                "gradient_mean_right",
                "gradient_mean_asymmetry",
                "rs_minus_cb",
                "rs_minus_cb__left",
                "rs_minus_cb__right",
            ]
        ]

    primary = metrics(left, right)
    supplementary, pairs = (None, None)
    if include_9v9:
        pairs = homotopic_pairs(atlas, members)
        supplementary = metrics(
            pairs.left_index_1based.to_numpy(int) - 1,
            pairs.right_index_1based.to_numpy(int) - 1,
        )
        prep.require(
            supplementary.sub_id.equals(primary.sub_id),
            "Homotopic participant order changed",
        )
    flow = None
    if "sample_flow" in paths:
        flow = prep.read_table(paths["sample_flow"])
        prep.require(
            flow.iloc[-1]["stage"] == "right_handed_final"
            and int(flow.iloc[-1]["n"]) == len(sample),
            "Sample flow does not match prepared sample",
        )
    out = prep.new_output(config_path, output_dir)
    primary.to_csv(out / "subject_metrics.csv", index=False)
    members.to_csv(out / "language_network_members.csv", index=False)
    prepared_runs.to_csv(out / "retained_runs.csv", index=False)
    if flow is not None:
        flow.to_csv(out / "sample_flow.csv", index=False)
    if supplementary is not None:
        supplementary.to_csv(out / "homotopic_subject_metrics.csv", index=False)
        pairs.to_csv(out / "homotopic_language_pairs.csv", index=False)
    prep.write_report(
        out,
        "extraction_report.json",
        config_path,
        {
            **paths,
            "extraction_script": Path(__file__).resolve(),
            "preparation_script": Path(prep.__file__).resolve(),
        },
        dict(
            status="extracted",
            sample_n=len(sample),
            sample_order="unchanged from prepared_sample",
            network_counts={"left": len(left), "right": len(right)},
            checks=limits,
            task="raw RS beta minus raw CB beta; equal-weight mean over retained runs",
            formula="left parcel mean minus right parcel mean",
            gradient="supplied aligned axis 0; not estimated here",
            standardization="none; performed inside each downstream model",
            beta=beta_qc,
            include_9v9=include_9v9,
            homotopic_scope="Fig2 supplementary network-level only",
        ),
    )
    print(f"Extracted {len(primary)} right-handed participants (L14-R9): {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/metrics"))
    args = parser.parse_args()
    settings = {"inputs": INPUTS, "extraction": {"include_9v9": True}}
    extract(settings, Path(__file__), args.output_dir)


if __name__ == "__main__":
    main()
