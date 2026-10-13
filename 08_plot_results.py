"""Draw the main and supplementary result panels from saved estimates."""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from PIL import Image

SETTINGS = {
    "schema_version": 1,
    "sample_n": 1167,
    "font_family": "Arial",
    "results": {
        "asymmetry": "results/asymmetry",
        "reading": "results/reading",
        "mediation": "results/mediation",
        "homotopic": "results/supplementary_9v9",
        "spectrum": "results/spectrum",
    },
}
OUTPUT_DIR = Path("figures/results")
DPI = 600
COLORS = {"connectivity": "#116496", "activation": "#116496", "reading": "#116496"}
STEMS = {
    "fig2a": "fig2a_connectivity_age",
    "fig2b": "fig2b_activation_age",
    "fig2c": "fig2c_model_coefficients",
    "fig2d": "fig2d_adjusted_association",
    "supp_fig1": "supp_fig1_gradient_variance",
    "fig3b": "fig3b_reading_age",
    "fig3c": "fig3c_connectivity_reading_coefficients",
    "fig3d": "fig3d_activation_reading_coefficients",
    "fig4b": "fig4b_indirect_three_groups",
    **{
        f"supp_fig2_9v9{p}": f"supp_fig2_9v9{p}_{name}"
        for p, name in zip(
            "abcd",
            (
                "connectivity_age",
                "activation_age",
                "model_coefficients",
                "adjusted_association",
            ),
        )
    },
}
NEW_TERMS = [
    "predictor",
    "age_linear",
    "age_quadratic",
    "predictor_x_age_linear",
    "predictor_x_age_quadratic",
]


def result_panels(results: dict, base: Path) -> dict:
    """Map the public analysis CSV interface onto fixed, independent panels."""
    unknown = set(results) - {
        "asymmetry",
        "reading",
        "mediation",
        "homotopic",
        "spectrum",
    }
    if unknown:
        raise ValueError(f"Unknown result directories: {sorted(unknown)}")
    panels = {}
    for name, directory in results.items():
        root = resolve_path(base, directory)

        def table(filename, model=None, columns=None):
            value = {"path": str(root / filename)}
            if model is not None:
                value["where"] = {"model": model}
            if columns:
                value["columns"] = columns
            return value

        if name in ("asymmetry", "homotopic"):
            prefix = "fig2" if name == "asymmetry" else "supp_fig2_9v9"
            for letter, model in (("a", "connectivity_age"), ("b", "activation_age")):
                panels[prefix + letter] = {
                    "scatter": table(
                        "age_scatter.csv", model, {"x": "age_years", "y": "observed"}
                    ),
                    "curve": table("age_curves.csv", model, {"x": "age_years"}),
                }
            scatter_letter, coefficient_letter = ("d", "c")
            panels[prefix + scatter_letter] = {
                "scatter": table(
                    "association_scatter.csv",
                    columns={"x": "predictor_residual", "y": "outcome_residual"},
                ),
                "curve": table("association_curve.csv"),
            }
            panels[prefix + coefficient_letter] = {
                "coefficients": table("coefficients.csv", "connectivity_activation"),
                "terms": NEW_TERMS,
            }
        elif name == "reading":
            panels["fig3b"] = {
                "scatter": table(
                    "age_scatter.csv",
                    "reading_age",
                    {"x": "age_years", "y": "observed"},
                ),
                "curve": table("age_curves.csv", "reading_age", {"x": "age_years"}),
            }
            for letter, model in (
                ("c", "connectivity_reading"),
                ("d", "activation_reading"),
            ):
                panels["fig3" + letter] = {
                    "coefficients": table("coefficients.csv", model),
                    "terms": NEW_TERMS,
                }
        elif name == "mediation":
            panels["fig4b"] = {"groups": table("age_group_indirect.csv")}
        else:
            panels["supp_fig1"] = {"percentages": table("participant_percentages.csv")}
    return panels


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def resolve_path(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def load_table(spec, base: Path, inputs: dict) -> pd.DataFrame:
    """Load and select a CSV without interpreting or recalculating statistics."""
    spec = {"path": spec} if isinstance(spec, str) else spec
    path = resolve_path(base, spec["path"])
    inputs[str(path)] = sha256(path)
    header = pd.read_csv(path, nrows=0).columns
    identifier = spec.get("columns", {}).get("sub_id", "sub_id")
    frame = pd.read_csv(
        path,
        float_precision="round_trip",
        dtype={identifier: str} if identifier in header else None,
    )
    for key, value in spec.get("where", {}).items():
        if key not in frame:
            raise ValueError(f"Missing selector column {key}: {path}")
        frame = frame.loc[frame[key].eq(value)]
    mapping = spec.get("columns", {})
    frame = frame.rename(
        columns={source: canonical for canonical, source in mapping.items()}
    )
    if frame.columns.duplicated().any():
        raise ValueError(f"Column mapping creates duplicates: {path}")
    for canonical, alias in {
        "ci_low": "ci95_lower",
        "ci_high": "ci95_upper",
        "p": "p_raw",
        "estimate": "beta",
    }.items():
        if canonical not in frame and alias in frame:
            frame = frame.rename(columns={alias: canonical})
    if frame.empty:
        raise ValueError(f"Empty table selection: {path}")
    return frame.reset_index(drop=True)


def numeric(frame: pd.DataFrame, columns: list[str]) -> None:
    if not set(columns).issubset(frame):
        raise ValueError(f"Missing columns: {sorted(set(columns) - set(frame))}")
    values = frame[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite values in {columns}")


def check_intervals(frame: pd.DataFrame) -> None:
    numeric(frame, ["estimate", "ci_low", "ci_high"])
    if (frame.ci_low > frame.ci_high).any():
        raise ValueError("Confidence interval bounds are reversed")


def boolean_mask(values: pd.Series) -> np.ndarray:
    lookup = {"true": True, "false": False, "1": True, "0": False}
    normalized = values.astype(str).str.lower()
    if not normalized.isin(lookup).all():
        raise ValueError(
            "visible must contain true/false or 1/0 without missing values"
        )
    return normalized.map(lookup).to_numpy(dtype=bool)


def apply_style(font: str) -> str:
    font_path = font_manager.findfont(font, fallback_to_default=False)
    mpl.rcParams.update(
        {
            "font.family": font,
            "font.size": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.linewidth": 0.7,
            "text.color": "#262626",
            "axes.labelcolor": "#262626",
            "axes.edgecolor": "#333333",
            "axes.grid": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    return font_path


def new_axes(kind: str):
    width, height, left, bottom, axis_width = {
        "scatter": (8.8, 8.0, 1.6, 1.15, 6.95),
        "coefficients": (8.8, 9.5, 1.6, 2.65, 6.95),
        "groups": (5.6, 8.0, 2.05, 1.15, 3.5),
    }[kind]
    pixels = [round(cm / 2.54 * DPI) for cm in (width, height)]
    fig = plt.figure(figsize=[np.nextafter(px / DPI, np.inf) for px in pixels], dpi=DPI)
    ax = fig.add_axes((left / width, bottom / height, axis_width / width, 6.5 / height))
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3, width=0.7, pad=4)
    return (fig, ax, (width, height))


def save_panel(fig, ax, target: Path, size_cm: tuple) -> dict:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    hidden = set()
    for axis in (ax.xaxis, ax.yaxis):
        lo, hi = sorted(axis.get_view_interval())
        for tick in [*axis.get_major_ticks(), *axis.get_minor_ticks()]:
            if not lo <= tick.get_loc() <= hi:
                hidden.update((id(tick.label1), id(tick.label2)))
    text = []
    for item in fig.findobj(match=mpl.text.Text):
        if not item.get_visible() or not item.get_text() or id(item) in hidden:
            continue
        box = item.get_window_extent(renderer)
        if not (
            0 <= box.x0 < box.x1 <= fig.bbox.width
            and 0 <= box.y0 < box.y1 <= fig.bbox.height
        ):
            raise ValueError(f"Text exceeds panel bounds: {item.get_text()}")
        text.append(
            {
                "text": item.get_text(),
                "bounds_px": list(box.extents),
                "font_size_pt": item.get_fontsize(),
                "font_family": item.get_fontfamily(),
            }
        )
    pixels = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    if not (np.all(pixels[[0, -1]] == 255) and np.all(pixels[:, [0, -1]] == 255)):
        raise ValueError("Panel content touches the canvas edge")
    Image.fromarray(pixels).save(target, compression="tiff_lzw", dpi=(DPI, DPI))
    result = dict(
        file=target.name,
        sha256=sha256(target),
        size_cm=list(size_cm),
        pixels=[pixels.shape[1], pixels.shape[0]],
        dpi=DPI,
        text=text,
        xlim=list(ax.get_xlim()),
        ylim=list(ax.get_ylim()),
        axes_bounds_px=list(ax.get_window_extent(renderer).bounds),
    )
    plt.close(fig)
    with Image.open(target) as image:
        if (
            image.mode != "RGB"
            or image.n_frames != 1
            or image.tag_v2[259] != 5
            or (image.size != tuple((round(cm / 2.54 * DPI) for cm in size_cm)))
            or (image.info["dpi"] != (DPI, DPI))
        ):
            raise RuntimeError("TIFF format validation failed")
    return result


def render_scatter(
    key: str, spec: dict, base: Path, out: Path, inputs: dict, n: int
) -> dict:
    scatter = load_table(spec["scatter"], base, inputs)
    curve = load_table(spec["curve"], base, inputs)
    numeric(scatter, ["x", "y"])
    check_intervals(curve)
    numeric(curve, ["x"])
    if (
        len(scatter) != n
        or "sub_id" not in scatter
        or scatter.sub_id.isna().any()
        or (not scatter.sub_id.is_unique)
    ):
        raise ValueError(f"{key}: scatter must contain {n} unique participant IDs")
    if len(curve) < 2 or not (np.diff(curve.x) > 0).all():
        raise ValueError(f"{key}: curve x must be strictly increasing")
    if key in ("fig2a", "fig3b") and "visible" not in scatter:
        raise ValueError(f"{key}: supply the previously established display mask")
    visible = (
        boolean_mask(scatter.visible)
        if "visible" in scatter
        else np.ones(n, dtype=bool)
    )
    reading = key == "fig3b"
    adjusted = key in {"fig2d", "supp_fig2_9v9d"}
    activation = not reading and key.endswith("b")
    color = COLORS[
        "reading" if reading else "activation" if activation else "connectivity"
    ]
    xlim = (-4, 4) if adjusted else (6, 16)
    ylim = (
        (50, 100)
        if reading
        else (-4, 4) if adjusted else (-0.3, 0.7) if activation else (-3, 5)
    )
    if not adjusted and (
        curve.x.min() < scatter.x.min() - 1e-10
        or curve.x.max() > scatter.x.max() + 1e-10
    ):
        raise ValueError(f"{key}: age curve extrapolates beyond the sample")
    visible = (
        visible
        & scatter.x.between(*xlim).to_numpy()
        & scatter.y.between(*ylim).to_numpy()
    )
    fig, ax, size = new_axes("scatter")
    shown = scatter.loc[visible]
    ax.scatter(
        shown.x, shown.y, s=8, color=color, alpha=0.24, edgecolors="none", zorder=1
    )
    ax.fill_between(
        curve.x,
        curve.ci_low,
        curve.ci_high,
        color=color,
        alpha=0.16,
        linewidth=0,
        zorder=2,
    )
    ax.plot(curve.x, curve.estimate, color=color, linewidth=1.7, zorder=3)
    ax.set(
        xlim=xlim,
        ylim=ylim,
        xlabel="Adjusted connectivity asymmetry" if adjusted else "Age (years)",
        ylabel=(
            "Accuracy (%)"
            if reading
            else (
                "Adjusted activation asymmetry"
                if adjusted
                else "Activation asymmetry" if activation else "Connectivity asymmetry"
            )
        ),
    )
    ax.set_xticks(np.arange(-4, 5, 2) if adjusted else np.arange(6, 17, 2))
    ax.set_yticks(
        np.arange(50, 101, 10)
        if reading
        else (
            np.arange(-4, 5, 2)
            if adjusted
            else [-0.3, -0.1, 0.1, 0.3, 0.5, 0.7] if activation else np.arange(-3, 6, 2)
        )
    )
    if reading:
        ax.tick_params(pad=2)
    ax.xaxis.labelpad = ax.yaxis.labelpad = 3
    stem = STEMS[key]
    scatter.to_csv(
        out / "source_data" / (stem + "_scatter.csv"), index=False, float_format="%.17g"
    )
    curve.to_csv(
        out / "source_data" / (stem + "_curve.csv"), index=False, float_format="%.17g"
    )
    return {
        **save_panel(fig, ax, out / (stem + ".tif"), size),
        "sample_n": n,
        "visible_n": int(visible.sum()),
        "hidden_ids": scatter.loc[~visible, "sub_id"].tolist(),
        "color": color,
        "residuals_restandardized": False,
        "peak_marker": False,
    }


def draw_intervals(ax, frame: pd.DataFrame, color: str) -> None:
    x = np.arange(len(frame))
    ax.bar(x, frame.estimate, width=0.58, color=color, linewidth=0)
    ax.vlines(x, frame.ci_low, frame.ci_high, color="#444444", linewidth=1.4)
    ax.plot(
        x,
        frame.ci_low,
        linestyle="none",
        marker="_",
        markersize=6,
        markeredgewidth=1.4,
        color="#444444",
    )
    ax.plot(
        x,
        frame.ci_high,
        linestyle="none",
        marker="_",
        markersize=6,
        markeredgewidth=1.4,
        color="#444444",
    )
    ax.axhline(0, color="#AAAAAA", linewidth=0.6, zorder=0)


def render_coefficients(
    key: str, spec: dict, base: Path, out: Path, inputs: dict, n: int
) -> dict:
    table = load_table(spec["coefficients"], base, inputs)
    predictor = "Activation" if key == "fig3d" else "Connectivity"
    term_prefix = "neural" if key.startswith("fig3") else "connectivity"
    old_terms = [
        term_prefix,
        "age_linear",
        "age_quadratic",
        term_prefix + "_age_linear",
        term_prefix + "_age_quadratic",
    ]
    terms = spec.get(
        "terms",
        (
            NEW_TERMS
            if "term" in table and table.term.eq("predictor").any()
            else old_terms
        ),
    )
    if (
        len(terms) != 5
        or len(set(terms)) != 5
        or "term" not in table
        or table.term.duplicated().any()
    ):
        raise ValueError(
            f"{key}: select one model with five distinct coefficient terms"
        )
    if not set(terms).issubset(table.term):
        raise ValueError(
            f"{key}: missing required terms {terms}; set the terms mapping explicitly"
        )
    table = table.set_index("term").loc[terms].reset_index().dropna(axis=1, how="all")
    check_intervals(table)
    numeric(table, ["p_corrected"])
    if "correction" not in table or not table.correction.eq("bonferroni").all():
        raise ValueError(f"{key}: require explicitly Bonferroni-corrected coefficients")
    if not table.p_corrected.between(0, 1).all() or (
        "n" in table and (not table.n.eq(n).all())
    ):
        raise ValueError(f"{key}: invalid corrected p values or sample size")
    fig, ax, size = new_axes("coefficients")
    color = COLORS[predictor.lower()]
    draw_intervals(ax, table, color)
    labels = [
        predictor + "\nasymmetry",
        "Age",
        "Age²",
        predictor + "\n× age",
        predictor + "\n× age²",
    ]
    ax.set_xticks(np.arange(5), labels, rotation=40, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", length=3, pad=5, labelsize=12)
    lower, upper, step, offset = (
        (-0.4, 0.6, 0.2, 0.015) if key.startswith("fig3") else (-0.2, 0.3, 0.1, 0.008)
    )
    needed_low = (
        min(float(table.ci_low.min()), float(table.estimate.min())) - 4 * offset
    )
    needed_high = (
        max(float(table.ci_high.max()), float(table.estimate.max())) + 4 * offset
    )
    if key.startswith("supp_"):
        lower = min(lower, np.floor(needed_low / step) * step)
        upper = max(upper, np.ceil(needed_high / step) * step)
    elif needed_low < lower or needed_high > upper:
        raise ValueError(
            f"{key}: coefficient intervals exceed the established display range"
        )
    ax.set(xlim=(-0.65, 4.65), ylim=(lower, upper), ylabel="Coef.")
    ax.set_yticks(np.arange(lower, upper + step / 2, step))
    ax.yaxis.labelpad = 3
    stars = []
    for i, row in enumerate(table.itertuples()):
        mark = (
            "***"
            if row.p_corrected < 0.001
            else (
                "**"
                if row.p_corrected < 0.01
                else "*" if row.p_corrected < 0.05 else ""
            )
        )
        stars.append(mark)
        if mark:
            positive = row.estimate >= 0
            ax.text(
                i,
                row.ci_high + offset if positive else row.ci_low - offset,
                mark,
                va="bottom" if positive else "top",
                ha="center",
                fontsize=12,
            )
    stem = STEMS[key]
    table.to_csv(
        out / "source_data" / (stem + ".csv"), index=False, float_format="%.17g"
    )
    return {
        **save_panel(fig, ax, out / (stem + ".tif"), size),
        "sample_n": n,
        "color": color,
        "stars": stars,
        "coefficients": table.to_dict(orient="records"),
    }


def render_groups(spec: dict, base: Path, out: Path, inputs: dict, n: int) -> dict:
    table = load_table(spec["groups"], base, inputs)
    labels = ["6-8", "9-11", "12-15"]
    if (
        "age_group" not in table
        or len(table) != 3
        or set(table.age_group) != set(labels)
    ):
        raise ValueError(
            "Fig4B requires the three observed-age groups 6-8, 9-11, 12-15"
        )
    table = table.set_index("age_group").loc[labels].reset_index()
    check_intervals(table)
    numeric(table, ["n"])
    if (
        (table.n <= 0).any()
        or not np.equal(table.n, np.floor(table.n)).all()
        or table.n.sum() != n
        or ("model_n" in table and (not table.model_n.eq(n).all()))
    ):
        raise ValueError("Fig4B group counts do not match the model sample")
    fig, ax, size = new_axes("groups")
    draw_intervals(ax, table, COLORS["connectivity"])
    ax.set_xticks(np.arange(3), [label.replace("-", "–") for label in labels])
    step = 0.025
    lower = (
        np.floor(min(0, float(table.ci_low.min()), float(table.estimate.min())) / step)
        * step
    )
    upper = (
        np.ceil(max(0, float(table.ci_high.max()), float(table.estimate.max())) / step)
        * step
    )
    if upper == lower:
        upper += step
    ax.set(
        xlim=(-0.45, 2.55),
        ylim=(lower, upper),
        xlabel="Age group (years)",
        ylabel="Indirect association",
    )
    ax.set_yticks(np.arange(lower, upper + step / 2, step))
    ax.tick_params(pad=3)
    ax.xaxis.labelpad = ax.yaxis.labelpad = 2
    stem = STEMS["fig4b"]
    table.to_csv(
        out / "source_data" / (stem + ".csv"), index=False, float_format="%.17g"
    )
    return {
        **save_panel(fig, ax, out / (stem + ".tif"), size),
        "sample_n": n,
        "groups": table.to_dict(orient="records"),
        "no_group_comparisons": True,
    }


def render_spectrum(spec, base, out, inputs, n):
    table = load_table(spec["percentages"], base, inputs)
    numeric(table, ["gradient", "percent"])
    if len(table) != 10 * n or table.duplicated(["sub_id", "gradient"]).any():
        raise ValueError(
            "Expected one percentage per participant and first-ten component"
        )
    if (
        set(table.gradient) != set(range(1, 11))
        or not table.groupby("gradient").size().eq(n).all()
    ):
        raise ValueError("Incomplete gradient spectrum")
    if not np.allclose(table.groupby("sub_id").percent.sum(), 100):
        raise ValueError("Gradient percentages must use the first-ten denominator")
    size = (16.5, 9.0)
    pixels = [round(cm / 2.54 * DPI) for cm in size]
    fig = plt.figure(figsize=[np.nextafter(px / DPI, np.inf) for px in pixels], dpi=DPI)
    ax = fig.add_axes((1.6 / size[0], 1.3 / size[1], 14.5 / size[0], 7.2 / size[1]))
    ax.spines[["top", "right"]].set_visible(False)
    rng = np.random.default_rng(42)
    values = [
        table.loc[table.gradient.eq(i), "percent"].to_numpy() for i in range(1, 11)
    ]
    for i, y in enumerate(values, 1):
        ax.scatter(
            i + rng.uniform(-0.22, 0.22, len(y)),
            y,
            s=6,
            alpha=0.24,
            color=COLORS["connectivity"],
            edgecolors="none",
        )
    ax.boxplot(
        values,
        positions=np.arange(1, 11),
        widths=0.5,
        showfliers=False,
        boxprops={"color": "black", "linewidth": 1.4},
        whiskerprops={"color": "black", "linewidth": 1.4},
        capprops={"color": "black", "linewidth": 1.4},
        medianprops={"color": "black", "linewidth": 1.7},
    )
    ax.scatter(
        np.arange(1, 11),
        [v.mean() for v in values],
        marker="D",
        color="black",
        s=24,
        zorder=4,
    )
    ax.set(
        xlim=(0.5, 10.5),
        ylim=(0, 60),
        xlabel="Gradient",
        ylabel="Variance explained (%)",
    )
    table.to_csv(out / "source_data" / f"{STEMS['supp_fig1']}.csv", index=False)
    return {
        **save_panel(fig, ax, out / f"{STEMS['supp_fig1']}.tif", size),
        "sample_n": n,
        "observations_above_axis_limit": int((table.percent > 60).sum()),
        "denominator": "first ten diffusion eigenvalues; all observations retained in summaries",
    }


def run(output_dir: str) -> dict:
    config_path = Path(__file__).resolve()
    base = Path.cwd()
    config = SETTINGS
    if config.get("schema_version") != 1:
        raise ValueError("Expected schema_version=1")
    if "results" in config and "panels" in config:
        raise ValueError("Use results directory mode or explicit panels, not both")
    panels = (
        result_panels(config["results"], base)
        if "results" in config
        else config.get("panels", {})
    )
    if "select_panels" in config:
        selected = config["select_panels"]
        if (
            not isinstance(selected, list)
            or len(set(selected)) != len(selected)
            or set(selected) - set(panels)
        ):
            raise ValueError(
                "select_panels must be distinct keys available in this configuration"
            )
        panels = {key: panels[key] for key in selected}
    if not panels or set(panels) - set(STEMS):
        raise ValueError(f"Select panels from {list(STEMS)}")
    n = config.get("sample_n", 1167)
    if not isinstance(n, int) or n < 1:
        raise ValueError("sample_n must be a positive integer")
    out = resolve_path(base, output_dir)
    if out == Path(__file__).resolve().parent:
        raise ValueError("Choose an output subdirectory, not the code directory itself")
    if out.exists() and any(out.iterdir()):
        raise ValueError(
            "Use a new or empty output directory; existing results are not overwritten"
        )
    font = config.get("font_family", "Arial")
    if "font_file" in config:
        font_manager.fontManager.addfont(str(resolve_path(base, config["font_file"])))
    font_path = apply_style(font)
    out.mkdir(parents=True, exist_ok=True)
    (out / "source_data").mkdir()
    inputs = {str(config_path): sha256(config_path)}
    results = {}
    for key, spec in panels.items():
        if key == "supp_fig1":
            results[key] = render_spectrum(spec, base, out, inputs, n)
        elif key == "fig4b":
            results[key] = render_groups(spec, base, out, inputs, n)
        elif key in ("fig2c", "fig3c", "fig3d", "supp_fig2_9v9c"):
            results[key] = render_coefficients(key, spec, base, out, inputs, n)
        else:
            results[key] = render_scatter(key, spec, base, out, inputs, n)
    for path, expected in inputs.items():
        if sha256(Path(path)) != expected:
            raise RuntimeError(f"Input changed during rendering: {path}")
    metadata = dict(
        schema_version=1,
        panels=results,
        input_sha256=inputs,
        model_refits=0,
        bootstrap_reruns=0,
        fdr_recalculations=0,
        font={"family": font, "sha256": sha256(Path(font_path))},
        script_sha256=sha256(Path(__file__)),
        software={
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": mpl.__version__,
            "pillow": Image.__version__,
        },
    )
    (out / "metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    result = run(args.output_dir)
    print(json.dumps({"completed": list(result["panels"])}))


if __name__ == "__main__":
    main()

