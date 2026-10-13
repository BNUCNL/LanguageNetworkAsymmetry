"""Prepare the study sample, reading data, retained runs, and head motion."""

from __future__ import annotations
import argparse
import hashlib
import io
import json
import platform
import re
import tarfile
from pathlib import Path
import numpy as np
import pandas as pd

INPUTS = {
    "demographics": "data/interim/demographics.csv",
    "behavior": "data/interim/behavior_full.csv",
    "sites": "data/interim/site_info.csv",
    "brain_index": "data/processed/brain_subject_index.csv",
    "left_handed": "data/processed/ccbd_left_handed_subjects.csv",
    "motion_run_level": "data/processed/ccbd_motion_fd_run_level.csv",
    "motion_subject_level": "data/processed/ccbd_motion_fd_subject_level.csv",
    "task_beta_run_level": "data/raw/task_fmri_mmp360/single_condition_beta_2026-08-31/task_cl_beta_mmp360_run_level.csv",
    "task_beta_subject_runmean": "data/raw/task_fmri_mmp360/single_condition_beta_2026-08-31/task_cl_beta_mmp360_subject_runmean.csv",
    "task_spm_audit": "data/raw/task_fmri_mmp360/single_condition_beta_2026-08-31/spm_design_audit.tsv",
}
BETA_INPUTS = {
    "participants": "../BIDS_DATA/MRI_cl_fmri_prep_20251024/participants.tsv",
    "dataset_root": "../BIDS_DATA/MRI_cl_fmri_prep_20251024/v1",
    "atlas_left": "../Downloads/cortex_ColeAnticevic_NetPartition_GlasserParcels_L.label.gii",
    "atlas_right": "../Downloads/cortex_ColeAnticevic_NetPartition_GlasserParcels_R.label.gii",
    "label_key": "external/atlases/ColeAnticevicNetPartition/CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR_LabelKey.txt",
}
AGE, GRADE, SEX = ("Age_continuous", "grade_num", "sex_num")
SEX_LABEL, STAGE = ("性别", "学段")
REGION = "站点流域"
READING = "三分钟快速阅读_总正确率（%）"
ID_ZH, SITE_ZH = ("被试编号", "入库站点")
REGIONS = [f"region_{i}" for i in range(1, 361)]
CONDITIONS = ["LS", "RS", "CB", "DS", "fix"]
SITE_NAMES = [
    "北京师范大学",
    "天津师范大学",
    "河南师范大学",
    "河南省儿童医院郑州儿童医院",
    "广州市妇女儿童医疗中心",
    "陕西师范大学",
    "东南大学",
    "中国科学院心理研究所",
    "华中师范大学",
    "四川师范大学",
    "山东师范大学",
    "杭州师范大学",
    "浙江大学",
    "贵州师范大学",
]
SAMPLE_COLUMNS = [
    "sub_id",
    SEX_LABEL,
    STAGE,
    AGE,
    GRADE,
    SEX,
    "site",
    REGION,
    "site_code",
    "array_row",
    READING,
    "task_run_n",
    "rest_motion",
    "task_motion_frame_n",
    "task_motion_run_n",
    "task_motion",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(config_path, value):
    return Path(value).expanduser().resolve()


def input_paths(config, config_path, required, optional=()):
    section = config.get("inputs", {})
    missing = sorted(set(required) - section.keys())
    require(not missing, f"Missing config.inputs keys: {missing}")
    paths = {
        key: resolve_path(config_path, section[key])
        for key in (*required, *optional)
        if section.get(key) is not None
    }
    for key, path in paths.items():
        require(path.is_file(), f"Missing input {key}: {path}")
    return paths


def new_output(config_path, value):
    path = resolve_path(config_path, value)
    require(
        not path == Path(__file__).resolve().parent,
        "Choose an output subdirectory, not the code directory itself",
    )
    require(not path.exists(), f"Output directory already exists: {path}")
    path.mkdir(parents=True)
    return path


def read_table(path, **kwargs):
    if Path(path).suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, dtype=str, **kwargs)
    return pd.read_csv(
        path,
        sep="\t" if Path(path).suffix.lower() in {".tsv", ".txt"} else ",",
        dtype=str,
        encoding="utf-8-sig",
        **kwargs,
    )


def ids(frame, column, label):
    require(column in frame, f"{label}: missing ID column {column}")
    frame = frame.rename(columns={column: "sub_id"}).copy()
    frame["sub_id"] = frame.sub_id.astype("string").str.strip()
    require(
        frame.sub_id.notna().all() and frame.sub_id.ne("").all(),
        f"{label}: missing or blank identifiers",
    )
    return frame


def unique(frame, keys, label):
    require(not frame.duplicated(keys).any(), f"{label}: duplicate keys {keys}")


def numeric(series):
    return pd.to_numeric(
        series.astype("string")
        .str.strip()
        .str.replace("%", "", regex=False)
        .replace({"": pd.NA, "-": pd.NA, "NA": pd.NA, "NaN": pd.NA, "None": pd.NA}),
        errors="coerce",
    )


def integers(series, label, minimum=0):
    values = pd.to_numeric(series, errors="raise").to_numpy(dtype=float)
    require(
        np.isfinite(values).all()
        and (values >= minimum).all()
        and np.equal(values, np.floor(values)).all(),
        f"{label}: invalid integer values",
    )
    return values.astype(np.int64)


def finite(frame, columns, label):
    require(
        np.isfinite(frame[columns].to_numpy(float)).all(), f"{label}: nonfinite values"
    )


def column(config, table, role, default):
    return config.get("columns", {}).get(table, {}).get(role, default)


def checks(config):
    result = {"expected_n": 1167, "left_handed_n": 178, "excluded_left_handed_n": 43}
    result.update(config.get("checks", {}))
    for key in ("expected_n", "left_handed_n", "excluded_left_handed_n"):
        require(
            isinstance(result[key], int) and result[key] >= 0, f"Invalid checks.{key}"
        )
    return result


def left_handed_ids(path, config):
    default = ID_ZH if path.suffix.lower() in {".xlsx", ".xls"} else "sub_id"
    frame = ids(
        read_table(path), column(config, "left_handed", "id", default), "handedness"
    )
    unique(frame, ["sub_id"], "handedness")
    require(
        len(frame) == checks(config)["left_handed_n"], "Left-handed list count mismatch"
    )
    if "handedness" in frame:
        require(
            frame.handedness.eq("left").all(),
            "Handedness file must contain only left-handed IDs",
        )
    return frame[["sub_id"]].assign(handedness="left")


def brain_index(path):
    frame = ids(read_table(path), "sub_id", "brain index")
    unique(frame, ["sub_id"], "brain index")
    frame["array_row"] = integers(frame.array_row, "array_row")
    require(
        np.array_equal(frame.array_row, np.arange(len(frame))),
        "Brain index rows must map one-to-one to array positions 0..N-1",
    )
    return frame[["sub_id", "array_row"]]


def beta_tables(paths):
    runs = ids(read_table(paths["task_beta_run_level"]), "subID", "beta runs")
    require(
        [c for c in runs if c.startswith("region_")] == REGIONS,
        "Beta regions must be ordered region_1..region_360",
    )
    runs["run"] = integers(runs.run, "beta run", 1)
    unique(runs, ["sub_id", "run", "condition"], "beta runs")
    require(
        set(runs.condition) == set(CONDITIONS),
        "Beta export must retain all five conditions",
    )
    require(
        runs.groupby(["sub_id", "run"]).condition.nunique().eq(5).all(),
        "Incomplete condition set in beta export",
    )
    runs = pd.concat([runs.drop(columns=REGIONS), runs[REGIONS].astype(float)], axis=1)
    finite(runs, REGIONS, "beta runs")
    audit = ids(read_table(paths["task_spm_audit"]), "subID", "SPM audit")
    audit["run"] = integers(audit.run, "SPM audit run", 1)
    unique(audit, ["sub_id", "run"], "SPM audit")
    retained = audit.loc[audit.status.eq("exported"), ["sub_id", "run"]]
    retained = retained.sort_values(["sub_id", "run"]).reset_index(drop=True)
    observed = (
        runs[["sub_id", "run"]]
        .drop_duplicates()
        .sort_values(["sub_id", "run"])
        .reset_index(drop=True)
    )
    require(
        observed.equals(retained), "Beta run keys differ from exported SPM run keys"
    )
    means = (
        runs.groupby(["sub_id", "condition"], sort=True)[REGIONS]
        .mean()
        .copy()
        .reset_index()
    )
    means["n_runs"] = runs.groupby(["sub_id", "condition"], sort=True).size().to_numpy()
    max_error = 0.0
    if "task_beta_subject_runmean" in paths:
        saved = ids(
            read_table(paths["task_beta_subject_runmean"]), "subID", "beta means"
        )
        require(
            [c for c in saved if c.startswith("region_")] == REGIONS,
            "Invalid runmean ROI columns",
        )
        unique(saved, ["sub_id", "condition"], "beta means")
        saved = saved.sort_values(["sub_id", "condition"]).reset_index(drop=True)
        require(
            saved[["sub_id", "condition"]].equals(means[["sub_id", "condition"]]),
            "Runmean and run-level subject/condition keys differ",
        )
        saved = pd.concat(
            [saved.drop(columns=REGIONS), saved[REGIONS].astype(float)], axis=1
        )
        finite(saved, REGIONS, "beta means")
        saved["n_runs"] = integers(saved.n_runs, "beta n_runs", 1)
        require(np.array_equal(saved.n_runs, means.n_runs), "Beta run counts differ")
        max_error = float(
            np.max(np.abs(saved[REGIONS].to_numpy() - means[REGIONS].to_numpy()))
        )
        require(max_error < 1e-10, "Saved beta means do not reproduce retained runs")
        means = saved
    return (
        means,
        retained,
        {
            "run_n": len(retained),
            "subject_n": retained.sub_id.nunique(),
            "runmean_max_abs_error": max_error,
            "scale": "raw beta",
            "required_conditions": CONDITIONS,
        },
    )


def fd_archive(path, task):
    pattern = re.compile(
        "^sub-([^_]+)_task-(rest|CL)_run-([0-9]+)_desc-confounds_timeseries_fd\\.tsv$"
    )
    rows = []
    with tarfile.open(path, "r:*") as archive:
        for member in archive:
            if not member.isfile():
                continue
            match = pattern.fullmatch(Path(member.name).name)
            require(
                match is not None and match[2] == task,
                f"Unexpected FD member: {member.name}",
            )
            stream = archive.extractfile(member)
            require(stream is not None, "Unreadable FD member")
            table = pd.read_csv(
                io.BytesIO(stream.read()), sep="\t", na_values=["n/a", "N/A"]
            )
            require(
                table.columns.tolist() == ["framewise_displacement"],
                "Invalid FD columns",
            )
            values = pd.to_numeric(table.iloc[:, 0], errors="raise").to_numpy(float)
            require(not np.isinf(values).any(), "Infinite FD value")
            valid = values[np.isfinite(values)]
            require(
                len(valid) > 0 and (valid >= 0).all(), "Missing or negative FD values"
            )
            rows.append(
                dict(
                    sub_id=match[1],
                    task=task,
                    run=int(match[3]),
                    n_fd_finite=len(valid),
                    mean_fd=float(valid.mean()),
                )
            )
    require(bool(rows), "Empty FD archive")
    return pd.DataFrame(rows)


def motion_tables(paths, retained):
    archives = "rest_fd_archive" in paths or "task_fd_archive" in paths
    require(
        not (archives and "motion_run_level" in paths),
        "Specify FD archives or run CSV, not both",
    )
    if archives:
        require(
            "rest_fd_archive" in paths and "task_fd_archive" in paths,
            "Both FD archives are required",
        )
        runs = pd.concat(
            [
                fd_archive(paths["rest_fd_archive"], "rest"),
                fd_archive(paths["task_fd_archive"], "CL"),
            ],
            ignore_index=True,
        )
    else:
        require("motion_run_level" in paths, "Missing motion_run_level or FD archives")
        runs = ids(read_table(paths["motion_run_level"]), "sub_id", "motion")
    runs = runs[["sub_id", "task", "run", "n_fd_finite", "mean_fd"]].copy()
    runs["run"] = integers(runs.run, "motion run", 1)
    runs["n_fd_finite"] = integers(runs.n_fd_finite, "FD frame count", 1)
    runs["mean_fd"] = pd.to_numeric(runs.mean_fd, errors="raise")
    finite(runs, ["mean_fd"], "motion")
    require(
        runs.mean_fd.ge(0).all() and set(runs.task) == {"rest", "CL"},
        "Invalid FD values/tasks",
    )
    unique(runs, ["sub_id", "task", "run"], "motion")

    def aggregate(table, prefix):
        table = table.assign(weighted=table.mean_fd * table.n_fd_finite)
        frame = table.groupby("sub_id", sort=True).agg(
            weighted=("weighted", "sum"),
            frames=("n_fd_finite", "sum"),
            run_n=("run", "nunique"),
        )
        frame[f"{prefix}_motion"] = frame.weighted / frame.frames
        return (
            frame.drop(columns="weighted")
            .rename(
                columns={
                    "frames": f"{prefix}_motion_frame_n",
                    "run_n": f"{prefix}_motion_run_n",
                }
            )
            .reset_index()
        )

    rest = aggregate(runs.loc[runs.task.eq("rest")], "rest")
    if "motion_subject_level" in paths:
        saved = ids(
            read_table(paths["motion_subject_level"]), "sub_id", "motion subjects"
        )
        unique(saved, ["sub_id"], "motion subjects")
        saved["rest_motion"] = numeric(saved.rest_motion)
        saved = saved.dropna(subset=["rest_motion"]).set_index("sub_id")
        require(
            set(saved.index) == set(rest.sub_id), "Rest FD subject coverage differs"
        )
        values = saved.loc[rest.sub_id, "rest_motion"].to_numpy(float)
        require(
            np.allclose(values, rest.rest_motion, atol=1e-12, rtol=0),
            "Rest FD aggregation mismatch",
        )
        rest["rest_motion"] = values
    linked = retained.merge(
        runs.loc[runs.task.eq("CL")].drop(columns="task"),
        on=["sub_id", "run"],
        how="left",
        validate="one_to_one",
    )
    finite(linked, ["n_fd_finite", "mean_fd"], "Retained task runs require FD")
    return (runs, rest, aggregate(linked, "task"))


def write_report(out, name, config_path, paths, details):
    report = {
        "schema_version": 1,
        "script_sha256": sha256(config_path),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "inputs": {
            key: {"path": str(path), "sha256": sha256(path)}
            for key, path in paths.items()
        },
        "random_seed": None,
        "bootstrap_count": 0,
        "statistical_models_fitted": 0,
        **details,
        "outputs": {
            p.name: {"sha256": sha256(p)} for p in sorted(out.iterdir()) if p.is_file()
        },
    }
    (out / name).write_text(
        json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    )


def prepare(config, config_path, output_dir):
    paths = input_paths(
        config,
        config_path,
        [
            "demographics",
            "behavior",
            "sites",
            "brain_index",
            "left_handed",
            "task_beta_run_level",
            "task_spm_audit",
        ],
        [
            "task_beta_subject_runmean",
            "motion_run_level",
            "motion_subject_level",
            "rest_fd_archive",
            "task_fd_archive",
        ],
    )
    limits = checks(config)
    demo = ids(
        read_table(paths["demographics"]),
        column(config, "demographics", "id", "subID"),
        "demographics",
    )
    renames = {
        column(config, "demographics", role, default): default
        for role, default in [
            ("age", AGE),
            ("grade", GRADE),
            ("sex", SEX),
            ("sex_label", SEX_LABEL),
            ("stage", STAGE),
        ]
    }
    demo = demo.rename(columns=renames)
    unique(demo, ["sub_id"], "demographics")
    for col in (AGE, GRADE, SEX):
        demo[col] = numeric(demo[col])
    for col in (SEX_LABEL, STAGE):
        if col not in demo:
            demo[col] = pd.NA
    demo = demo[["sub_id", SEX_LABEL, STAGE, AGE, GRADE, SEX]]
    demo = demo.loc[demo[GRADE].between(1, 9) & demo[AGE].between(6.0, 16.5)].copy()
    require(demo[GRADE].dropna().mod(1).eq(0).all(), "Grade must be an integer")
    site = ids(
        read_table(paths["sites"]), column(config, "sites", "id", ID_ZH), "sites"
    )
    site = site.rename(
        columns={
            column(config, "sites", "site", SITE_ZH): "site",
            column(config, "sites", "region", REGION): REGION,
        }
    )
    unique(site, ["sub_id"], "sites")
    codebook = config.get("site_codebook", dict(zip(SITE_NAMES, range(1, 15))))
    require(
        isinstance(codebook, dict) and bool(codebook),
        "site_codebook must be a nonempty object",
    )
    codes = integers(pd.Series(list(codebook.values())), "site_codebook", 1)
    require(len(set(codes)) == len(codes), "Site codes must be one-to-one")
    require(set(site.site.dropna()).issubset(codebook), "Unmapped site name")
    site["site_code"] = site.site.map(codebook)
    if REGION not in site:
        site[REGION] = pd.NA
    site = site[["sub_id", "site", REGION, "site_code"]]
    behavior = ids(
        read_table(paths["behavior"]),
        column(config, "behavior", "id", ID_ZH),
        "behavior",
    )
    behavior = behavior.rename(
        columns={column(config, "behavior", "reading", READING): READING}
    )
    unique(behavior, ["sub_id"], "behavior")
    behavior[READING] = numeric(behavior[READING])
    means, retained, beta_qc = beta_tables(paths)
    runs, rest, task_fd = motion_tables(paths, retained)
    task = means.loc[means.condition.eq("RS"), ["sub_id", "n_runs"]].rename(
        columns={"n_runs": "task_run_n"}
    )
    handedness = left_handed_ids(paths["left_handed"], config)
    flow = [{"stage": "eligible_demographics", "n": len(demo)}]
    excluded = []
    data = demo
    for stage, table in [
        ("after_site", site),
        ("after_gradient_index", brain_index(paths["brain_index"])),
        ("after_three_minute_behavior", behavior[["sub_id", READING]]),
        ("after_single_condition_task", task),
        ("after_rest_motion", rest[["sub_id", "rest_motion"]]),
        ("after_task_motion", task_fd),
    ]:
        lost = data.loc[~data.sub_id.isin(table.sub_id), ["sub_id"]].assign(stage=stage)
        excluded.append(lost)
        data = data.merge(table, on="sub_id", how="inner", validate="one_to_one")
        flow.append(dict(stage=stage, n=len(data)))
    required = [
        AGE,
        GRADE,
        SEX,
        "site_code",
        "array_row",
        READING,
        "rest_motion",
        "task_motion",
        "task_run_n",
    ]
    missing = data[required].isna().any(axis=1)
    excluded.append(
        data.loc[missing, ["sub_id"]].assign(stage="complete_case_before_handedness")
    )
    data = data.loc[~missing].copy()
    flow.append(dict(stage="complete_case_before_handedness", n=len(data)))
    finite(data, required, "complete sample")
    require(set(data[SEX]).issubset({0, 1}), "Sex must be coded female=0, male=1")
    labeled = data[SEX_LABEL].notna()
    if labeled.any():
        mapping = config.get("sex_label_codebook", {"女": 0, "男": 1})
        require(
            np.array_equal(
                data.loc[labeled, SEX_LABEL].map(mapping).to_numpy(float),
                data.loc[labeled, SEX].to_numpy(float),
            ),
            "Sex labels and codes disagree",
        )
    require(
        data[READING].between(0, 100).all(),
        "Reading accuracy must be in 0..100 percent",
    )
    require(
        np.array_equal(data.task_run_n, data.task_motion_run_n),
        "Beta and task FD run counts differ",
    )
    left = data.sub_id.isin(handedness.sub_id)
    require(
        int(left.sum()) == limits["excluded_left_handed_n"],
        "Excluded left-handed count mismatch",
    )
    excluded.append(data.loc[left, ["sub_id"]].assign(stage="right_handed_final"))
    data = data.loc[~left, SAMPLE_COLUMNS].sort_values("sub_id").reset_index(drop=True)
    require(
        len(data) == limits["expected_n"] and len(data) > 0,
        "Final right-handed sample size mismatch",
    )
    flow.append(dict(stage="right_handed_final", n=len(data)))
    for col in (
        GRADE,
        SEX,
        "site_code",
        "array_row",
        "task_run_n",
        "task_motion_run_n",
        "task_motion_frame_n",
    ):
        data[col] = integers(data[col], col)
    out = new_output(config_path, output_dir)
    data.to_csv(out / "prepared_sample.csv", index=False)
    pd.DataFrame(flow).to_csv(out / "sample_flow.csv", index=False)
    pd.concat(excluded, ignore_index=True).to_csv(
        out / "sample_exclusions.csv", index=False
    )
    retained.loc[retained.sub_id.isin(data.sub_id)].to_csv(
        out / "retained_runs.csv", index=False
    )
    handedness.to_csv(out / "ccbd_left_handed_subjects.csv", index=False)
    runs.to_csv(out / "ccbd_motion_fd_run_level.csv", index=False)
    summary = (
        data[[AGE, READING, "rest_motion", "task_motion"]].astype(float).describe().T
    )
    summary.to_csv(out / "sample_numeric_summary.csv", index_label="variable")
    counts = [
        dict(variable=col, level=str(key), n=int(value))
        for col in (SEX, GRADE, "site_code")
        for key, value in data[col].value_counts().sort_index().items()
    ]
    pd.DataFrame(counts).to_csv(out / "sample_categorical_summary.csv", index=False)
    write_report(
        out,
        "preparation_report.json",
        config_path,
        {**paths, "preparation_script": Path(__file__).resolve()},
        dict(
            status="prepared_pending_gradient_extraction",
            sample_n=len(data),
            checks=limits,
            eligibility={
                "age": [6.0, 16.5],
                "grade": [1, 9],
                "handedness": "right only",
            },
            site_codebook=codebook,
            sample_order="sub_id ascending",
            beta=beta_qc,
            task_motion="finite-frame weighted FD over exported beta runs only",
            rest_motion="finite-frame weighted FD over all available rest runs",
            reading=READING,
            standardization="none; performed inside each downstream model",
        ),
    )
    print(f"Prepared {len(data)} right-handed participants: {out}")


def extract_beta(config, config_path, output_dir):
    import nibabel as nib
    import scipy
    from scipy.io import loadmat

    spec = config.get("beta_extraction", {})
    names = ["participants", "dataset_root", "atlas_left", "atlas_right", "label_key"]
    require(all((key in spec for key in names)), f"beta_extraction requires {names}")
    paths = {key: resolve_path(config_path, spec[key]) for key in names}
    require(paths["dataset_root"].is_dir(), "Missing first-level dataset root")

    def gifti(path):
        image = nib.load(str(path))
        require(len(image.darrays) == 1, f"Expected one GIFTI array: {path}")
        values = np.asarray(image.darrays[0].data, dtype=float)
        require(values.shape == (32492,), f"Expected fsLR-32k vertex vector: {path}")
        return values

    labels = []
    for key, first, last in [("atlas_left", 1, 180), ("atlas_right", 181, 360)]:
        values = gifti(paths[key])
        require(
            np.isfinite(values).all()
            and (values >= 0).all()
            and (values == np.floor(values)).all(),
            "Invalid atlas labels",
        )
        require(
            np.array_equal(np.unique(values[values > 0]), np.arange(first, last + 1)),
            "Invalid atlas label range",
        )
        labels.append(values.astype(int))
    labels = np.concatenate(labels)
    atlas = read_table(paths["label_key"]).iloc[:360].copy()
    require(
        np.array_equal(integers(atlas.INDEX, "atlas INDEX", 1), np.arange(1, 361)),
        "Invalid atlas index order",
    )
    require(
        atlas.HEMISPHERE.tolist() == ["L"] * 180 + ["R"] * 180,
        "Invalid hemisphere order",
    )
    participants = ids(
        read_table(paths["participants"]),
        spec.get("participant_column", ID_ZH),
        "participants",
    )
    participants["sub_id"] = participants.sub_id.str.removeprefix("sub-")
    require(
        participants.sub_id.str.fullmatch("[A-Za-z0-9-]+").all(),
        "Unsafe participant identifier",
    )
    unique(participants, ["sub_id"], "participants")
    rows, audit, issues, beta_audit = ([], [], [], [])
    design_pattern = re.compile("Sn\\((\\d+)\\)\\s*(.*?)\\*bf\\((\\d+)\\)", re.I)
    target = {c.casefold(): c for c in CONDITIONS}
    for subject in participants.sub_id:
        parent = paths["dataset_root"] / f"sub-{subject}" / "func" / "cl_fmri_1stlevel"
        pattern = f"sub-{subject}_task-CL_run-*_space-fsLR_desc-taskFriston_custom-smooth_first-level"
        directories = sorted((p for p in parent.glob(pattern) if p.is_dir()))
        seen = set()
        if not directories:
            issues.append(
                dict(subID=subject, run="", message="No matching first-level run")
            )
        for directory in directories:
            match = re.search("_run-([0-9]+)_", directory.name)
            require(match is not None, f"Invalid run directory: {directory.name}")
            run = int(match[1])
            require(run > 0 and run not in seen, "Duplicate or invalid first-level run")
            seen.add(run)
            spm_path = directory / "SPM.mat"
            record = dict(
                subID=subject, run=run, status="skipped", spm_path=str(spm_path)
            )
            pending, pending_audit = ([], [])
            try:
                record["spm_sha256"] = sha256(spm_path)
                spm = loadmat(spm_path, simplify_cells=True)["SPM"]
                design = [
                    str(value).strip()
                    for value in np.asarray(spm["xX"]["name"], dtype=object).reshape(-1)
                ]
                record["design_columns_json"] = json.dumps(design)
                mapping = {}
                for index, name in enumerate(design, start=1):
                    found = design_pattern.search(name)
                    if not found or int(found[3]) != 1:
                        continue
                    key = re.sub("\\s+", "", found[2]).casefold()
                    if key not in target:
                        continue
                    require(
                        int(found[1]) == 1 and target[key] not in mapping,
                        "Ambiguous/session-mismatched condition column",
                    )
                    mapping[target[key]] = index
                require(
                    set(mapping) == set(CONDITIONS), "Missing target bf(1) condition"
                )
                record["condition_to_beta_json"] = json.dumps(mapping, sort_keys=True)
                for condition in CONDITIONS:
                    index = mapping[condition]
                    beta_paths = [
                        directory / f"beta_{index:04d}_hemi-{h}.func.gii"
                        for h in ("L", "R")
                    ]
                    values = np.concatenate([gifti(path) for path in beta_paths])
                    valid = np.isfinite(values) & (labels > 0)
                    counts = np.bincount(labels[valid], minlength=361)[1:]
                    require((counts > 0).all(), "Parcel without finite beta vertices")
                    means = (
                        np.bincount(
                            labels[valid], weights=values[valid], minlength=361
                        )[1:]
                        / counts
                    )
                    pending.append(
                        dict(
                            subID=subject,
                            run=run,
                            condition=condition,
                            beta_index=index,
                            design_column_name=design[index - 1],
                            **dict(zip(REGIONS, means)),
                        )
                    )
                    for hemi, path in zip(("L", "R"), beta_paths):
                        pending_audit.append(
                            dict(
                                subID=subject,
                                run=run,
                                condition=condition,
                                hemisphere=hemi,
                                path=str(path),
                                sha256=sha256(path),
                            )
                        )
            except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
                issues.append(dict(subID=subject, run=run, message=str(exc)))
            else:
                rows.extend(pending)
                beta_audit.extend(pending_audit)
                record["status"] = "exported"
            audit.append(record)
    require(bool(rows), "No complete runs exported; check first-level data and atlas")
    runs = pd.DataFrame(rows).sort_values(["subID", "run", "condition"])
    means = (
        runs.groupby(["subID", "condition"], sort=True)[REGIONS].mean().reset_index()
    )
    means.insert(
        2, "n_runs", runs.groupby(["subID", "condition"], sort=True).size().to_numpy()
    )
    atlas["vertex_count"] = np.bincount(labels, minlength=361)[1:]
    out = new_output(config_path, output_dir)
    runs.to_csv(out / "task_cl_beta_mmp360_run_level.csv", index=False)
    means.to_csv(out / "task_cl_beta_mmp360_subject_runmean.csv", index=False)
    pd.DataFrame(audit).to_csv(out / "spm_design_audit.tsv", sep="\t", index=False)
    pd.DataFrame(issues, columns=["subID", "run", "message"]).to_csv(
        out / "extraction_issues.tsv", sep="\t", index=False
    )
    pd.DataFrame(beta_audit).to_csv(out / "beta_input_audit.tsv", sep="\t", index=False)
    atlas.to_csv(out / "roi_metadata.tsv", sep="\t", index=False)
    file_inputs = {k: p for k, p in paths.items() if k != "dataset_root"}
    write_report(
        out,
        "extraction_manifest.json",
        config_path,
        {**file_inputs, "preparation_script": Path(__file__).resolve()},
        dict(
            status="extracted",
            subject_n=runs.subID.nunique(),
            exported_run_n=len(runs) // 5,
            issue_n=len(issues),
            dataset_root=str(paths["dataset_root"]),
            conditions=CONDITIONS,
            rule="All five conditions and both hemispheres required per run; finite-vertex parcel mean",
            aggregation="unweighted within-subject arithmetic mean across retained runs",
            extra_software={"nibabel": nib.__version__, "scipy": scipy.__version__},
        ),
    )
    print(f"Exported {len(runs) // 5} complete beta runs: {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["prepare", "extract-beta"], nargs="?", default="prepare"
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or Path(
        "results/task_betas" if args.command == "extract-beta" else "results/prepared"
    )
    settings = {"inputs": INPUTS, "beta_extraction": BETA_INPUTS}
    (prepare if args.command == "prepare" else extract_beta)(
        settings, Path(__file__), output
    )


if __name__ == "__main__":
    main()

