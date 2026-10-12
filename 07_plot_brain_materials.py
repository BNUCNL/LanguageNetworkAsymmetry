"""Draw the Fig1 brain maps, FC matrix, and atlas illustrations."""

from __future__ import annotations
import re
import nibabel as nib
import nilearn
from nilearn import plotting
from matplotlib.colors import ListedColormap
from bs4 import BeautifulSoup

VIEWS = (("L", "lateral"), ("R", "lateral"), ("L", "medial"), ("R", "medial"))
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import matplotlib

matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from PIL import Image

SETTINGS = {
    "schema_version": 1,
    "sample_n": 1167,
    "atlas_dir": "external/atlases/ColeAnticevicNetPartition",
    "materials": {
        "gradient": {
            "mean_csv": "manuscript/reports/figure1_concept_20260925/displayed_gradient_values.csv",
            "index_column": "parcel_index",
            "value_column": "mean_principal_gradient",
            "sample_n": 1167,
        },
        "activation": {
            "mean_csv": "manuscript/reports/figure1_integrated_20260925/activation_mean_parcels.csv",
            "index_column": "INDEX",
            "value_column": "mean_rs_minus_cb_raw_beta",
            "sample_n": 1167,
        },
        "fc": {
            "mean_matrix": "manuscript/reports/fig1_fc_matrix_20260925/mean_fc_original_order.npy",
            "sample_n": 1167,
        },
    },
    "atlas_illustrations": {
        "glasser_colors": "external/atlases/HCPMMP1_original_colors/BALSA_2xZq_public_label_table.html",
        "networks": True,
        "language_members": True,
    },
}
OUTPUT_DIR = Path("figures/fig1")
DPI = 600
KEY_NAME = (
    "CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR_LabelKey.txt"
)
NETWORK_ORDER = [
    "Visual1",
    "Visual2",
    "Somatomotor",
    "Cingulo-Opercular",
    "Dorsal-Attention",
    "Language",
    "Frontoparietal",
    "Auditory",
    "Default",
    "Posterior-Multimodal",
    "Ventral-Multimodal",
    "Orbito-Affective",
]
STEMS = {
    "gradient": "fig1_connectivity_gradient_lateral",
    "activation": "fig1_rs_cb_activation_lateral",
    "fc": "fig1_functional_connectivity_matrix",
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def resolve_path(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def input_path(base: Path, value: str, inputs: dict, expected=None) -> Path:
    path = resolve_path(base, value)
    actual = sha256(path)
    if expected is not None and actual != expected:
        raise ValueError(f"Input hash mismatch: {path}")
    inputs[str(path)] = actual
    return path


def load_atlas(directory: Path, inputs: dict) -> pd.DataFrame:
    key = input_path(directory, KEY_NAME, inputs)
    atlas = pd.read_csv(key, sep="\t").iloc[:360].copy()
    if atlas.INDEX.tolist() != list(range(1, 361)):
        raise ValueError("Atlas must use original 1-based cortical INDEX order")
    if not (
        atlas.iloc[:180].HEMISPHERE.eq("L").all()
        and atlas.iloc[180:].HEMISPHERE.eq("R").all()
    ):
        raise ValueError("Atlas hemisphere order must be left 1-180, right 181-360")
    members = atlas.loc[atlas.NETWORK.eq("Language")]
    if members.HEMISPHERE.value_counts().to_dict() != {"L": 14, "R": 9}:
        raise ValueError("Original Language membership must be L14/R9")
    return atlas


def load_sample(config: dict, base: Path, inputs: dict) -> pd.DataFrame:
    path = input_path(base, config["sample"], inputs)
    sample = pd.read_csv(path, dtype={"sub_id": str}, float_precision="round_trip")
    if (
        "sub_id" not in sample
        or sample.sub_id.isna().any()
        or (not sample.sub_id.is_unique)
        or (len(sample) != config.get("sample_n", 1167))
    ):
        raise ValueError(
            "Sample must contain the expected number of unique participant IDs"
        )
    return sample


def check_asymmetry(
    values: np.ndarray,
    sample: pd.DataFrame,
    atlas: pd.DataFrame,
    column: str,
    target: Path,
) -> dict:
    members = atlas.loc[atlas.NETWORK.eq("Language")]
    left = members.loc[members.HEMISPHERE.eq("L"), "INDEX"].to_numpy(int) - 1
    right = members.loc[members.HEMISPHERE.eq("R"), "INDEX"].to_numpy(int) - 1
    contrast = values[:, left].mean(axis=1) - values[:, right].mean(axis=1)
    record = pd.DataFrame({"sub_id": sample.sub_id, "left14_minus_right9": contrast})
    error = None
    if column in sample:
        reference = sample[column].to_numpy(float)
        np.testing.assert_allclose(contrast, reference, rtol=0, atol=1e-10)
        error = float(np.max(np.abs(contrast - reference)))
        record["reference"] = reference
    record.to_csv(target, index=False, float_format="%.17g")
    return {
        "reference_column": column if column in sample else None,
        "max_abs_error": error,
    }


def extract_gradient(
    spec: dict,
    sample: pd.DataFrame,
    atlas: pd.DataFrame,
    base: Path,
    inputs: dict,
    out: Path,
) -> tuple[np.ndarray, dict]:
    source = input_path(base, spec["array"], inputs, spec.get("sha256"))
    index_path = input_path(base, spec["subject_index"], inputs)
    index = pd.read_csv(index_path, dtype={"sub_id": str})
    if (
        "sub_id" not in index
        or index.sub_id.isna().any()
        or (not index.sub_id.is_unique)
    ):
        raise ValueError("Gradient subject index must contain unique IDs")
    array = np.load(source, mmap_mode="r", allow_pickle=False)
    if (
        array.ndim not in (2, 3)
        or array.shape[0] != len(index)
        or array.shape[1] != 360
    ):
        raise ValueError("Gradient array and subject index dimensions disagree")
    rows = pd.Index(index.sub_id).get_indexer(sample.sub_id)
    if (rows < 0).any():
        raise ValueError("Sample participants are missing from the gradient array")
    if "array_row" in sample:
        np.testing.assert_array_equal(rows, sample.array_row.to_numpy())
    values = np.asarray(
        array[rows, :, 0] if array.ndim == 3 else array[rows], dtype=float
    )
    if not np.isfinite(values).all():
        raise ValueError("Non-finite aligned gradient coordinates")
    check = check_asymmetry(
        values,
        sample,
        atlas,
        "gradient_mean_asymmetry",
        out / "gradient_subject_check.csv",
    )
    return (
        values.mean(axis=0),
        {
            "source_mode": "participant_mean",
            "sample_n": len(sample),
            "gradient_component_zero_based": 0,
            "asymmetry_check": check,
        },
    )


def extract_activation(
    spec: dict,
    sample: pd.DataFrame,
    atlas: pd.DataFrame,
    base: Path,
    inputs: dict,
    out: Path,
) -> tuple[np.ndarray, dict]:
    beta_path = input_path(base, spec["run_betas"], inputs, spec.get("sha256"))
    retained_path = input_path(base, spec["retained_runs"], inputs)
    columns = [f"region_{i}" for i in range(1, 361)]
    betas = pd.read_csv(
        beta_path, dtype={"subID": str, "run": str}, float_precision="round_trip"
    )
    retained = pd.read_csv(
        retained_path,
        sep="\t" if retained_path.suffix == ".tsv" else ",",
        dtype={"subID": str, "sub_id": str, "run": str},
    )
    if "sub_id" in retained and "subID" not in retained:
        retained = retained.rename(columns={"sub_id": "subID"})
    if "status" in retained:
        retained = retained.loc[retained.status.eq("exported")]
    retained = retained.loc[retained.subID.isin(sample.sub_id), ["subID", "run"]]
    if (
        retained.isna().any().any()
        or retained.duplicated().any()
        or set(retained.subID) != set(sample.sub_id)
    ):
        raise ValueError("Invalid retained run list or incomplete sample coverage")
    counts = retained.groupby("subID").size().reindex(sample.sub_id)
    for name in ("task_run_n", "task_motion_run_n"):
        if name in sample:
            np.testing.assert_array_equal(counts.to_numpy(), sample[name].to_numpy())
    if [c for c in betas if c.startswith("region_")] != columns:
        raise ValueError("Beta parcels must be region_1 through region_360 in order")
    betas = betas.loc[
        betas.subID.isin(sample.sub_id) & betas.condition.isin(["RS", "CB"])
    ]
    if betas.duplicated(["subID", "run", "condition"]).any():
        raise ValueError("Duplicate participant/run/condition betas")
    stored = None
    if "runmean" in spec:
        stored = pd.read_csv(
            input_path(base, spec["runmean"], inputs),
            dtype={"subID": str},
            float_precision="round_trip",
        )
    maps = {}
    expected = set(map(tuple, retained.to_numpy()))
    for condition in ("RS", "CB"):
        frame = betas.loc[betas.condition.eq(condition)].merge(
            retained, on=["subID", "run"], validate="one_to_one"
        )
        if (
            set(zip(frame.subID, frame.run)) != expected
            or not np.isfinite(frame[columns].to_numpy(float)).all()
        ):
            raise ValueError(f"Missing or non-finite retained {condition} betas")
        if "design_column_name" in frame and (
            not frame.design_column_name.eq(f"Sn(1) {condition}*bf(1)").all()
        ):
            raise ValueError(f"Unexpected SPM design column for {condition}")
        maps[condition] = (
            frame.groupby("subID")[columns].mean().loc[sample.sub_id].to_numpy(float)
        )
        if stored is not None:
            reference = stored.loc[stored.condition.eq(condition)].set_index("subID")
            if not reference.index.is_unique:
                raise ValueError("Duplicate stored run means")
            np.testing.assert_allclose(
                maps[condition],
                reference.loc[sample.sub_id, columns],
                rtol=0,
                atol=1e-12,
            )
            if "n_runs" in reference:
                np.testing.assert_array_equal(
                    counts.to_numpy(), reference.loc[sample.sub_id, "n_runs"].to_numpy()
                )
    values = maps["RS"] - maps["CB"]
    check = check_asymmetry(
        values, sample, atlas, "rs_minus_cb", out / "activation_subject_check.csv"
    )
    retained.sort_values(["subID", "run"]).to_csv(
        out / "activation_retained_runs.csv", index=False
    )
    return (
        values.mean(axis=0),
        {
            "source_mode": "participant_mean",
            "sample_n": len(sample),
            "retained_run_n": len(retained),
            "task": "RS-CB raw beta",
            "run_weighting": "equal within participant",
            "asymmetry_check": check,
        },
    )


def read_mean(
    spec: dict, kind: str, base: Path, inputs: dict, n: int
) -> tuple[np.ndarray, dict]:
    if spec.get("sample_n") != n:
        raise ValueError("Precomputed means require an explicit matching sample_n")
    path = input_path(base, spec["mean_csv"], inputs, spec.get("sha256"))
    table = pd.read_csv(path, float_precision="round_trip")
    default_index, default_value = (
        ("parcel_index", "mean_principal_gradient")
        if kind == "gradient"
        else ("INDEX", "mean_rs_minus_cb_raw_beta")
    )
    index, value = (
        spec.get("index_column", default_index),
        spec.get("value_column", default_value),
    )
    if len(table) != 360 or sorted(table[index].tolist()) != list(range(1, 361)):
        raise ValueError("Mean table must contain exactly the 360 original parcel IDs")
    if "n" in table and (not table.n.eq(n).all()):
        raise ValueError("Precomputed mean table sample counts disagree")
    values = table.sort_values(index)[value].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite precomputed means")
    return (
        values,
        {
            "source_mode": "supplied_mean",
            "sample_n": n,
            "participant_data_verified": False,
        },
    )


def cortical_data(path: Path) -> dict:
    import nibabel as nib

    image = nib.load(path)
    values = np.asarray(image.dataobj)[0]
    result = {}
    for name, section, model in image.header.get_axis(1).iter_structures():
        if name in ("CIFTI_STRUCTURE_CORTEX_LEFT", "CIFTI_STRUCTURE_CORTEX_RIGHT"):
            if model.nvertices[name] != 32492:
                raise ValueError("Expected fsLR32k cortical vertices")
            full = np.zeros(32492)
            full[model.vertex] = values[section]
            result[name] = full
    if len(result) != 2:
        raise ValueError("Missing cortical hemisphere")
    return result


def surface_resources(atlas_dir: Path, atlas: pd.DataFrame, inputs: dict) -> dict:
    import nibabel as nib

    indices = cortical_data(
        input_path(atlas_dir, "Glasser360Indices_LR.dscalar.nii", inputs)
    )
    sulc = cortical_data(
        input_path(atlas_dir, "S1200.sulc_MSMAll.32k_fs_LR.dscalar.nii", inputs)
    )
    labels = cortical_data(
        input_path(
            atlas_dir,
            "CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR.dlabel.nii",
            inputs,
        )
    )
    resources = {}
    for hemi, side, first in (("L", "Left", 1), ("R", "Right", 181)):
        key = "CIFTI_STRUCTURE_CORTEX_" + side.upper()
        parcel = indices[key]
        np.testing.assert_array_equal(
            np.unique(parcel[parcel > 0]), np.arange(first, first + 180)
        )
        np.testing.assert_array_equal(parcel, labels[key])
        path = input_path(
            atlas_dir, f"S1200.{hemi}.inflated_MSMAll.32k_fs_LR.surf.gii", inputs
        )
        surface = nib.load(path)
        coords, faces = surface.agg_data()
        if surface.darrays[0].meta.get(
            "AnatomicalStructurePrimary"
        ) != "Cortex" + side or coords.shape != (32492, 3):
            raise ValueError("Incorrect surface hemisphere or geometry")
        if not np.isfinite(coords).all() or not np.isfinite(sulc[key]).all():
            raise ValueError("Non-finite surface geometry or shading")
        roi = atlas.loc[
            atlas.NETWORK.eq("Language") & atlas.HEMISPHERE.eq(hemi), "INDEX"
        ].tolist()
        resources[hemi] = dict(
            surface=(coords, faces),
            parcel_index=parcel.astype(int),
            background=sulc[key],
            roi_ids=roi,
        )
    return resources


def content_bbox(image: Image.Image) -> tuple:
    yy, xx = np.nonzero(np.any(np.asarray(image) < 250, axis=2))
    if not len(xx):
        raise ValueError("Surface render is blank")
    return (int(xx.min()), int(yy.min()), int(xx.max() + 1), int(yy.max() + 1))


def render_surface(
    values: np.ndarray, resources: dict, kind: str
) -> tuple[Image.Image, dict]:
    """Render original signed parcel means with the established lateral geometry."""
    from nilearn import plotting

    if values.shape != (360,) or not np.isfinite(values).all():
        raise ValueError("Surface values must contain 360 finite means")
    if kind == "gradient":
        limit = float(np.max(np.abs(values)))
        if limit <= 0:
            raise ValueError("All-zero gradient has no nondegenerate color scale")
        low, high = (-limit, limit)
        cmap = LinearSegmentedColormap.from_list(
            "white_to_manuscript_blue", ["#FFFFFF", "#116496"]
        )
    else:
        low, high = (-0.8, 0.4)
        cmap = LinearSegmentedColormap.from_list(
            "negative_gray_zero_white_positive_orange",
            [(0, "#AEB5BB"), ((0 - low) / (high - low), "#FFFFFF"), (1, "#B75A26")],
            N=1201,
        )
    if np.any((values < low) | (values > high)):
        raise ValueError(
            "Surface means exceed the established color bounds; do not silently clip"
        )
    width, height = (round(cm / 2.54 * DPI) for cm in (9, 3.3))
    padding, gutter = (
        max(2, round(0.025 / 2.54 * DPI)),
        max(2, round(0.035 / 2.54 * DPI)),
    )
    cell_width, cell_height = (
        (width - 2 * padding - gutter) // 2,
        height - 2 * padding,
    )
    side_pixels = int(np.ceil(max(cell_width, cell_height) * 1.5))
    rendered, views = ([], [])
    for hemi in ("L", "R"):
        resource = resources[hemi]
        parcel = resource["parcel_index"]
        vertices = np.full(32492, np.nan)
        valid = parcel > 0
        vertices[valid] = values[parcel[valid] - 1]
        fig = Figure(
            figsize=(side_pixels / DPI, side_pixels / DPI), dpi=DPI, facecolor="white"
        )
        canvas = FigureCanvasAgg(fig)
        try:
            ax = fig.add_axes((0, 0, 1, 1), projection="3d", facecolor="white")
            coords, faces = resource["surface"]
            mesh = (coords.copy(), faces)
            plotting.plot_surf_stat_map(
                mesh,
                vertices,
                bg_map=resource["background"],
                hemi="left" if hemi == "L" else "right",
                view="lateral",
                engine="matplotlib",
                cmap=cmap,
                vmin=low,
                vmax=high,
                symmetric_cbar=kind == "gradient",
                colorbar=False,
                threshold=None,
                avg_method="median",
                bg_on_data=True,
                alpha=1,
                axes=ax,
                figure=fig,
            )
            plotting.plot_surf_contours(
                mesh,
                parcel,
                levels=resource["roi_ids"],
                colors=["black"] * len(resource["roi_ids"]),
                legend=False,
                axes=ax,
                figure=fig,
            )
            ax.collections[0].set_linewidth(0.05)
            ax.set_proj_type("ortho")
            ax.set_xlim(float(mesh[0][:, 0].min()), float(mesh[0][:, 0].max()))
            ax.set_ylim(float(mesh[0][:, 1].min()), float(mesh[0][:, 1].max()))
            ax.set_zlim(float(mesh[0][:, 2].min()), float(mesh[0][:, 2].max()))
            ax.set_box_aspect(np.ptp(mesh[0], axis=0), zoom=1.0)
            ax.set_axis_off()
            canvas.draw()
            image = Image.fromarray(np.asarray(canvas.buffer_rgba()).copy()).convert(
                "RGB"
            )
            box = content_bbox(image)
            if min(box[0], box[1], image.width - box[2], image.height - box[3]) < 3:
                raise ValueError("Surface silhouette may be clipped")
            rendered.append(
                image.crop((box[0] - 2, box[1] - 2, box[2] + 2, box[3] + 2))
            )
            views.append(
                dict(
                    hemisphere=hemi,
                    view="lateral",
                    language_roi_n=len(resource["roi_ids"]),
                    source_not_clipped=True,
                )
            )
        finally:
            fig.clear()
    image = Image.new("RGB", (width, height), "white")
    scale = min(
        cell_width / max((im.width for im in rendered)),
        cell_height / max((im.height for im in rendered)),
    )
    for i, view in enumerate(rendered):
        view = view.resize(
            (round(view.width * scale), round(view.height * scale)),
            Image.Resampling.LANCZOS,
        )
        x = padding + i * (cell_width + gutter) + (cell_width - view.width) // 2
        y = padding + (cell_height - view.height) // 2
        image.paste(view, (x, y))
        black = int(np.count_nonzero(np.all(np.asarray(view) < 40, axis=2)))
        if black == 0:
            raise ValueError("Missing Language boundaries")
        views[i].update(
            paste_bbox_px=[x, y, x + view.width, y + view.height],
            black_outline_pixels=black,
        )
    return (
        image,
        dict(
            color_range=[low, high],
            views=views,
            size_cm=[9, 3.3],
            saturated_parcel_count=0,
            value_range=[float(values.min()), float(values.max())],
        ),
    )


def extract_fc(
    spec: dict, sample: pd.DataFrame, base: Path, inputs: dict
) -> np.ndarray:
    path = input_path(base, spec["edges"], inputs, spec.get("sha256"))
    selected, seen = (set(sample.sub_id), set())
    total = np.zeros(64620)
    with path.open(newline="") as stream:
        reader = csv.reader(stream)
        if next(reader)[1:] != [str(i) for i in range(64620)]:
            raise ValueError(
                "FC columns must encode the 64620 strict-upper-triangle edges"
            )
        for row in reader:
            sid = row[0]
            if sid not in selected:
                continue
            if sid in seen or len(row) != 64621:
                raise ValueError("Duplicate participant or malformed FC row")
            values = np.asarray(row[1:], dtype=float)
            if not np.isfinite(values).all():
                raise ValueError("Non-finite FC input")
            total += values
            seen.add(sid)
    if seen != selected:
        raise ValueError("FC input does not cover the complete selected sample")
    matrix = np.full((360, 360), np.nan)
    ix = np.triu_indices(360, 1)
    matrix[ix] = total / len(selected)
    matrix[ix[1], ix[0]] = matrix[ix]
    return matrix


def render_fc(
    matrix: np.ndarray, atlas: pd.DataFrame
) -> tuple[Image.Image, dict, np.ndarray, pd.DataFrame]:
    if matrix.shape != (360, 360) or not np.isnan(np.diag(matrix)).all():
        raise ValueError(
            "FC means require a 360 x 360 matrix with missing diagonal as NaN"
        )
    np.testing.assert_allclose(matrix, matrix.T, rtol=0, atol=0, equal_nan=True)
    edges = matrix[np.triu_indices(360, 1)]
    if not np.isfinite(edges).all() or set(atlas.NETWORK) != set(NETWORK_ORDER):
        raise ValueError("Invalid FC values or network labels")
    order = atlas.assign(
        network_order=atlas.NETWORK.map({v: i for i, v in enumerate(NETWORK_ORDER)})
    ).sort_values(["network_order", "INDEX"], kind="stable")
    perm = order.INDEX.to_numpy(int) - 1
    display = matrix[np.ix_(perm, perm)]
    limit = max(0.05, float(np.ceil(np.quantile(np.abs(edges), 0.95) / 0.05) * 0.05))
    cmap = LinearSegmentedColormap.from_list(
        "blue_white_orange", ["#116496", "#FFFFFF", "#B75A26"], N=1201
    )
    rgba = cmap(
        Normalize(vmin=-limit, vmax=limit, clip=True)(np.nan_to_num(display, nan=0)),
        bytes=True,
    )
    rgba[np.eye(360, dtype=bool)] = [255, 255, 255, 255]
    image = (
        Image.fromarray(rgba)
        .convert("RGB")
        .resize((1417, 1417), Image.Resampling.NEAREST)
    )
    return (
        image,
        dict(
            color_range=[-limit, limit],
            size_cm=[6, 6],
            diagonal="missing, displayed white",
            input_scale="stored FC scale, no Fisher or inverse-Fisher transformation",
            saturated_negative_edges=int((edges < -limit).sum()),
            saturated_positive_edges=int((edges > limit).sum()),
            value_range=[float(edges.min()), float(edges.max())],
        ),
        display,
        order,
    )


def save_images(image: Image.Image, out: Path, stem: str) -> dict:
    result = {}
    for extension in ("tif", "png"):
        target = out / (stem + "." + extension)
        image.save(
            target,
            dpi=(DPI, DPI),
            **{"compression": "tiff_lzw"} if extension == "tif" else {},
        )
        with Image.open(target) as stored:
            np.testing.assert_array_equal(np.asarray(stored), np.asarray(image))
            if stored.mode != "RGB" or (
                extension == "tif"
                and (stored.tag_v2[259] != 5 or stored.info["dpi"] != (DPI, DPI))
            ):
                raise ValueError("Image format validation failed")
        result[target.name] = sha256(target)
    return result


def render_gradient_brain(
    values: np.ndarray,
    members: pd.DataFrame,
    width_cm: float = 4.2,
    height_cm: float = 2.8,
    dpi: int = 600,
    *,
    outline_language: bool = True,
    lateral_only: bool = False,
    cmap="coolwarm",
    display_limit: float | None = None,
    display_range: tuple[float, float] | None = None,
    outline_parcels: list[int] | None = None,
    outline_groups: np.ndarray | None = None,
    resources,
) -> tuple[Image.Image, dict]:
    """Render parcel values with the manuscript surface geometry and boundaries."""
    if np.iscomplexobj(values):
        raise ValueError("values must be real gradient coordinates")
    values = np.asarray(values, dtype=float)
    if values.shape != (360,) or not np.isfinite(values).all():
        raise ValueError("values must have shape (360,) and contain only finite values")
    limit = (
        float(np.max(np.abs(values))) if display_limit is None else float(display_limit)
    )
    if limit == 0:
        raise ValueError(
            "An all-zero gradient has no nondegenerate symmetric color range"
        )
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError("display_limit must be finite and positive")
    if display_range is not None and display_limit is not None:
        raise ValueError("Specify display_range or display_limit, not both")
    low, high = (-limit, limit) if display_range is None else tuple(display_range)
    if not np.isfinite([low, high]).all() or low >= high:
        raise ValueError("display_range must contain two finite increasing bounds")
    dimensions = np.asarray([width_cm, height_cm, dpi], dtype=float)
    if not np.isfinite(dimensions).all() or np.any(dimensions <= 0):
        raise ValueError("width_cm, height_cm and dpi must be finite and positive")
    width, height = (round(float(cm) / 2.54 * dpi) for cm in (width_cm, height_cm))
    padding = max(2, round(0.025 / 2.54 * dpi))
    gutter = max(2, round(0.035 / 2.54 * dpi))
    views = VIEWS[:2] if lateral_only else VIEWS
    n_rows = len(views) // 2
    cell_width = (width - 2 * padding - gutter) // 2
    cell_height = (height - 2 * padding - (n_rows - 1) * gutter) // n_rows
    if min(cell_width, cell_height) < 32:
        raise ValueError("Requested image is too small for the cortical views")
    qc = {}
    if outline_parcels is not None:
        if any(
            (
                not isinstance(i, (int, np.integer)) or not 1 <= i <= 360
                for i in outline_parcels
            )
        ):
            raise ValueError("Outline parcels must be integer Glasser indices 1..360")
    if outline_groups is not None:
        outline_groups = np.asarray(outline_groups)
        if (
            outline_groups.shape != (360,)
            or not np.isfinite(outline_groups).all()
            or np.any(outline_groups <= 0)
            or np.any(outline_groups != np.floor(outline_groups))
        ):
            raise ValueError("Outline groups must be 360 positive integer category IDs")
        if outline_parcels is not None or outline_language:
            raise ValueError(
                "Group outlines cannot be combined with parcel or Language outlines"
            )
    side_pixels = int(np.ceil(max(cell_width, cell_height) * 1.5))
    rendered, view_qc = ([], [])
    for hemi, view in views:
        resource = resources[hemi]
        parcel_index = resource["parcel_index"]
        valid = parcel_index > 0
        vertex_values = np.full(32492, np.nan)
        vertex_values[valid] = values[parcel_index[valid] - 1]
        figure = Figure(
            figsize=(side_pixels / dpi, side_pixels / dpi), dpi=dpi, facecolor="white"
        )
        canvas = FigureCanvasAgg(figure)
        try:
            axes = figure.add_axes((0, 0, 1, 1), projection="3d", facecolor="white")
            coords, faces = resource["surface"]
            mesh = (coords.copy(), faces)
            plotting.plot_surf_stat_map(
                mesh,
                vertex_values,
                bg_map=resource["background"],
                hemi="left" if hemi == "L" else "right",
                view=view,
                engine="matplotlib",
                cmap=cmap,
                vmin=low,
                vmax=high,
                symmetric_cbar=display_range is None,
                colorbar=False,
                threshold=None,
                avg_method="median",
                bg_on_data=True,
                alpha=1,
                axes=axes,
                figure=figure,
            )
            outline_ids = (
                resource["roi_ids"].tolist()
                if outline_parcels is None
                else sorted(set(outline_parcels) & set(parcel_index[valid]))
            )
            contour_labels = parcel_index
            if outline_groups is not None:
                contour_labels = np.zeros_like(parcel_index)
                contour_labels[valid] = outline_groups[parcel_index[valid] - 1]
                outline_ids = np.unique(contour_labels[valid]).tolist()
            if (
                outline_language
                or outline_parcels is not None
                or outline_groups is not None
            ) and outline_ids:
                plotting.plot_surf_contours(
                    mesh,
                    contour_labels,
                    levels=outline_ids,
                    colors=["black"] * len(outline_ids),
                    legend=False,
                    axes=axes,
                    figure=figure,
                )
            axes.collections[0].set_linewidth(0.05)
            axes.set_proj_type("ortho")
            axes.set_xlim(
                *[float(v) for v in (mesh[0][:, 0].min(), mesh[0][:, 0].max())]
            )
            axes.set_ylim(
                *[float(v) for v in (mesh[0][:, 1].min(), mesh[0][:, 1].max())]
            )
            axes.set_zlim(
                *[float(v) for v in (mesh[0][:, 2].min(), mesh[0][:, 2].max())]
            )
            axes.set_box_aspect(np.ptp(mesh[0], axis=0), zoom=1.0)
            axes.set_axis_off()
            canvas.draw()
            image = Image.fromarray(np.asarray(canvas.buffer_rgba()).copy()).convert(
                "RGB"
            )
            box = content_bbox(image)
            if min(box[0], box[1], image.width - box[2], image.height - box[3]) < 3:
                raise RuntimeError(f"{hemi} {view} surface may be clipped")
            box = (box[0] - 2, box[1] - 2, box[2] + 2, box[3] + 2)
            rendered.append(image.crop(box))
            view_qc.append(
                {
                    "hemisphere": hemi,
                    "view": view,
                    "row": (len(rendered) - 1) // 2,
                    "column": (len(rendered) - 1) % 2,
                    "camera_elevation": float(axes.elev),
                    "camera_azimuth": float(axes.azim),
                    "source_content_bbox_px": list(box),
                    "source_canvas_size_px": list(image.size),
                    "source_not_clipped": True,
                }
            )
        finally:
            figure.clear()
    output = Image.new("RGB", (width, height), "white")
    scale = min(
        cell_width / max((im.width for im in rendered)),
        cell_height / max((im.height for im in rendered)),
    )
    for index, image in enumerate(rendered):
        image = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
        x = (
            padding
            + index % 2 * (cell_width + gutter)
            + (cell_width - image.width) // 2
        )
        y = (
            padding
            + index // 2 * (cell_height + gutter)
            + (cell_height - image.height) // 2
        )
        output.paste(image, (x, y))
        pixels = np.asarray(image)
        view_qc[index].update(
            {
                "paste_bbox_px": [x, y, x + image.width, y + image.height],
                "nonwhite_fraction": float(np.mean(np.any(pixels < 250, axis=2))),
                "black_outline_pixels": int(
                    np.count_nonzero(np.all(pixels < 40, axis=2))
                ),
            }
        )
    output.info["dpi"] = (float(dpi), float(dpi))
    qc.update(
        {
            "image_mode": output.mode,
            "size_px": list(output.size),
            "requested_size_cm": [float(width_cm), float(height_cm)],
            "dpi": float(dpi),
            "values_sha256_float64_le": hashlib.sha256(
                values.astype("<f8").tobytes()
            ).hexdigest(),
            "input_value_range": [float(values.min()), float(values.max())],
            "colormap": getattr(cmap, "name", str(cmap)),
            "color_range": [float(low), float(high)],
            "normalization": (
                "caller-specified common display range; values unchanged"
                if display_range is not None
                else (
                    "max(abs(values)); no clipping, z-scoring or sign flip"
                    if display_limit is None
                    else "caller-specified symmetric display limit; endpoint colors for out-of-range values; input unchanged"
                )
            ),
            "saturated_parcel_count": int(
                np.count_nonzero((values < low) | (values > high))
            ),
            "vertex_mapping": "values[Glasser_INDEX - 1]; medial wall masked",
            "surface": "S1200 inflated MSMAll fsLR32k",
            "sulc_role": "surface shading only; no structural analysis",
            "outline_method": (
                "black boundary-face band per categorical network"
                if outline_groups is not None
                else (
                    "black boundary-face band per specified Glasser parcel"
                    if outline_parcels is not None
                    else (
                        "black boundary-face band per original Language ROI"
                        if outline_language
                        else "none"
                    )
                )
            ),
            "language_outlined": outline_language and outline_parcels is None,
            "view_count": len(view_qc),
            "views": view_qc,
            "content_bbox_px": list(content_bbox(output)),
            "no_text_axes_arrows_or_colorbar": True,
            "software": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "nibabel": nib.__version__,
                "nilearn": nilearn.__version__,
                "matplotlib": matplotlib.__version__,
                "pillow": Image.__version__,
            },
        }
    )
    return (output, qc)


def original_glasser_colors(atlas, source):
    source = Path(source)
    provenance = {"sha256": sha256(source)}
    soup = BeautifulSoup(source.read_bytes(), "html.parser")
    records = []
    for row in soup.select("div.row"):
        cells = row.find_all("div", recursive=False)
        if len(cells) != 3 or "background-color" not in cells[1].get("style", ""):
            continue
        name = cells[2].get_text(strip=True)
        if not name.startswith(("L_", "R_")) or not name.endswith("_ROI"):
            continue
        match = re.fullmatch(
            "background-color:\\s*rgba\\(([^()]+)\\)\\s*;?", cells[1]["style"]
        )
        assert match is not None, name
        rgba = json.loads("[" + match.group(1) + "]")
        assert len(rgba) == 4 and all((0 <= channel <= 255 for channel in rgba))
        records.append(
            dict(
                source_key=int(cells[0].get_text(strip=True).rstrip(":")),
                GLASSERLABELNAME=name,
                **dict(zip(["RED", "GREEN", "BLUE", "ALPHA"], rgba)),
            )
        )
    colors = pd.DataFrame(records)
    assert len(colors) == 360 and colors.GLASSERLABELNAME.is_unique
    assert set(colors.GLASSERLABELNAME) == set(atlas.GLASSERLABELNAME)
    ordered = (
        atlas[["INDEX", "GLASSERLABELNAME"]]
        .merge(colors, on="GLASSERLABELNAME", validate="one_to_one")
        .sort_values("INDEX")
    )
    assert ordered.INDEX.tolist() == list(range(1, 361))
    assert ordered.source_key.tolist() == list(range(181, 361)) + list(range(1, 181))
    assert len(ordered[["RED", "GREEN", "BLUE", "ALPHA"]].drop_duplicates()) == 360
    return (ordered, source, provenance)


def render_atlas_materials(spec, base, atlas_dir, atlas, resources, out, inputs):
    members = atlas.loc[atlas.NETWORK.eq("Language")].copy()
    resources = {
        hemi: {**resource, "roi_ids": np.asarray(resource["roi_ids"], dtype=int)}
        for hemi, resource in resources.items()
    }
    jobs = []
    if spec.get("glasser_colors"):
        color_path = input_path(base, spec["glasser_colors"], inputs)
        colors, _, _ = original_glasser_colors(atlas, color_path)
        cmap = ListedColormap(
            colors[["RED", "GREEN", "BLUE", "ALPHA"]].to_numpy() / 255,
            name="Glasser2016_HCP_MMP10_original",
        )
        jobs.append(
            (
                "fig1_glasser360_lateral",
                np.arange(1, 361),
                cmap,
                (0.5, 360.5),
                list(range(1, 361)),
                None,
                False,
            )
        )
        colors.to_csv(out / "source_data/parcel_colors.csv", index=False)
    if spec.get("networks"):
        path = input_path(
            atlas_dir,
            "CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_netassignments_LR.dlabel.nii",
            inputs,
        )
        image = nib.load(path)
        palette = image.header.get_axis(0).label[0]
        values = atlas.NETWORKKEY.to_numpy(int)
        network_values = cortical_data(path)
        for hemi, side in [("L", "LEFT"), ("R", "RIGHT")]:
            index = resources[hemi]["parcel_index"]
            valid = index > 0
            np.testing.assert_array_equal(
                values[index[valid] - 1],
                network_values["CIFTI_STRUCTURE_CORTEX_" + side][valid],
            )
        cmap = ListedColormap(
            [palette[i][1] for i in range(1, 13)], name="Cole12_official"
        )
        jobs.append(
            (
                "fig1_cole12_networks_lateral",
                values,
                cmap,
                (0.5, 12.5),
                None,
                values,
                False,
            )
        )
    if spec.get("language_members"):
        left = members.loc[members.HEMISPHERE.eq("L")].copy()
        right = members.loc[members.HEMISPHERE.eq("R")].copy()
        left["base"] = left.GLASSERLABELNAME.str.removeprefix("L_")
        right["base"] = right.GLASSERLABELNAME.str.removeprefix("R_")
        pairs = left.merge(
            right, on="base", suffixes=("_left", "_right"), validate="one_to_one"
        )
        if len(pairs) != 9 or not np.array_equal(
            pairs.INDEX_right, pairs.INDEX_left + 180
        ):
            raise ValueError("Expected nine homologous language-network pairs")
        left_only = left.loc[~left.base.isin(right.base)]
        values = np.zeros(360, dtype=int)
        values[np.r_[pairs.INDEX_left, pairs.INDEX_right] - 1] = 1
        values[left_only.INDEX.to_numpy(int) - 1] = 2
        if (values == 1).sum() != 18 or (values == 2).sum() != 5:
            raise ValueError("Language membership does not match L14/R9")
        cmap = ListedColormap(
            ["#E4E7EA", "#116496", "#B75A26"], name="Language_membership_categories"
        )
        jobs.append(
            (
                "fig1_language_network_members_lateral",
                values,
                cmap,
                (-0.5, 2.5),
                None,
                None,
                True,
            )
        )
    results = {}
    for stem, values, cmap, bounds, parcels, groups, language in jobs:
        image, qc = render_gradient_brain(
            values,
            members,
            width_cm=9,
            height_cm=3.3,
            dpi=DPI,
            lateral_only=True,
            outline_language=language,
            outline_parcels=parcels,
            outline_groups=groups,
            cmap=cmap,
            display_range=bounds,
            resources=resources,
        )
        results[stem] = {
            **qc,
            "atlas_illustration": True,
            "outputs": save_images(image, out, stem),
        }
    return results


def run(output_dir: str) -> dict:
    config_path = Path(__file__).resolve()
    base = Path.cwd()
    config = SETTINGS
    if config.get("schema_version") != 1:
        raise ValueError("Expected schema_version=1")
    materials = config.get("materials", {})
    if (
        not materials
        and (not config.get("atlas_illustrations"))
        or set(materials) - set(STEMS)
    ):
        raise ValueError("Select gradient, activation, and/or fc materials")
    for kind, spec in materials.items():
        choices = (
            ("edges", "mean_matrix")
            if kind == "fc"
            else (
                ("array", "mean_csv")
                if kind == "gradient"
                else ("run_betas", "mean_csv")
            )
        )
        if sum((name in spec for name in choices)) != 1:
            raise ValueError(f"{kind}: supply exactly one of {choices}")
    out = resolve_path(base, output_dir)
    if out == Path(__file__).resolve().parent:
        raise ValueError("Choose an output subdirectory, not the code directory itself")
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new or empty output directory")
    n = config.get("sample_n", 1167)
    if not isinstance(n, int) or n < 1:
        raise ValueError("sample_n must be a positive integer")
    inputs = {str(config_path): sha256(config_path)}
    atlas_dir = resolve_path(base, config["atlas_dir"])
    atlas = load_atlas(atlas_dir, inputs)
    needs_sample = any(
        (
            "mean_csv" not in spec and "mean_matrix" not in spec
            for spec in materials.values()
        )
    )
    sample = load_sample(config, base, inputs) if needs_sample else None
    resources = (
        surface_resources(atlas_dir, atlas, inputs)
        if set(materials) & {"gradient", "activation"}
        or config.get("atlas_illustrations")
        else None
    )
    out.mkdir(parents=True, exist_ok=True)
    source = out / "source_data"
    source.mkdir()
    results = {}
    for kind, spec in materials.items():
        if kind == "fc":
            if "mean_matrix" in spec:
                if spec.get("sample_n") != n:
                    raise ValueError(
                        "Precomputed FC mean requires an explicit matching sample_n"
                    )
                matrix = np.load(
                    input_path(base, spec["mean_matrix"], inputs, spec.get("sha256")),
                    allow_pickle=False,
                )
                provenance = dict(
                    source_mode="supplied_mean",
                    sample_n=n,
                    participant_data_verified=False,
                )
            else:
                matrix = extract_fc(spec, sample, base, inputs)
                provenance = dict(source_mode="participant_mean", sample_n=n)
            image, qc, display, order = render_fc(matrix, atlas)
            np.save(source / "mean_fc_original_order.npy", matrix, allow_pickle=False)
            np.save(source / "mean_fc_display_order.npy", display, allow_pickle=False)
            order.to_csv(source / "parcel_display_order.csv", index=False)
        else:
            if "mean_csv" in spec:
                values, provenance = read_mean(spec, kind, base, inputs, n)
            else:
                extractor = (
                    extract_gradient if kind == "gradient" else extract_activation
                )
                values, provenance = extractor(
                    spec, sample, atlas, base, inputs, source
                )
            table = atlas[["INDEX", "HEMISPHERE", "NETWORK", "GLASSERLABELNAME"]].copy()
            table["n"], table["mean"] = (n, values)
            table.to_csv(
                source / (kind + "_mean_parcels.csv"), index=False, float_format="%.17g"
            )
            image, qc = render_surface(values, resources, kind)
        results[kind] = dict(
            **provenance,
            **qc,
            pixels=list(image.size),
            dpi=DPI,
            outputs=save_images(image, out, STEMS[kind]),
        )
    if config.get("atlas_illustrations"):
        results.update(
            render_atlas_materials(
                config["atlas_illustrations"],
                base,
                atlas_dir,
                atlas,
                resources,
                out,
                inputs,
            )
        )
    for path, expected in inputs.items():
        if sha256(Path(path)) != expected:
            raise RuntimeError(f"Input changed during rendering: {path}")
    software = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
        "pillow": Image.__version__,
    }
    if resources is not None:
        import nibabel
        import nilearn

        software.update(nibabel=nibabel.__version__, nilearn=nilearn.__version__)
    report = dict(
        schema_version=1,
        materials=results,
        input_sha256=inputs,
        script_sha256=sha256(Path(__file__)),
        model_refits=0,
        software=software,
    )
    (out / "metadata.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    result = run(args.output_dir)
    print(json.dumps({"completed": list(result["materials"])}))


if __name__ == "__main__":
    main()
