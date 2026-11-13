"""Summarize the reward study: how each setting changes the gaits and metrics.

Reads logs/study_results/runs/go2osc_study_<setting>__<run>.npz (from
study_eval.py) and compares each setting's seeds with the baseline's:

- gait change: total-variation distance between the setting's and the
  baseline's gait shares (pooled over seeds), averaged over the 20 cells;
  0 = same mix, 1 = no overlap. Its p-value comes from an exact permutation
  test: every way of splitting the pooled runs into two groups of the same
  sizes, so it says whether the change exceeds seed-to-seed variation.
- seed agreement: fraction of seed pairs with the same majority gait, per
  cell, averaged over cells.
- gait entropy: of the pooled shares [bits], averaged over cells.
- per-run metrics averaged over cells (tracking error, falls, peak force,
  load unevenness, turn rate, sideways drift): mean +- SD over seeds, with an
  exact permutation p-value for the difference from the baseline.
- convergence (if study_eval.py --checkpoint 1250 was run): gait distance
  between each run's 1250 and final checkpoints.

Writes study_summary.csv, study_metrics.png, study_gait_maps.png (majority
gait), study_gait_mix_maps.png (every gait's share) and study_gait_shares.png
to results/study/. With --wide, the same for the wide-range study
(go2osc_wide_*, 63 cells to 4.5 m/s and 6 Hz) in results/study/wide/.

Usage:
    uv run --frozen scripts/study_summary.py
    uv run --frozen scripts/study_summary.py --wide
"""

import argparse
import csv
import itertools
import re
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from grid_eval import (
    HOP_MAX_REAR_LAG,
    HOP_MAX_REAR_LEAD,
    gait_colors,
    observed_legend,
)
from gym.envs.go2.go2osc_config import STUDY_BASE_WEIGHTS, STUDY_SETTINGS, WIDE_SETTINGS
from gym.utils.helpers import GYM_ROOT_DIR

# Rebound by use_wide() for the wide-range study.
PREFIX = "go2osc_study_"
# Data (evaluation .npz, logs, caches) stays under logs/; everything meant to
# be looked at (figures, PDFs, videos, the summary table) goes to results/.
DATA = Path(GYM_ROOT_DIR) / "logs" / "study_results"
FIGURES = Path(GYM_ROOT_DIR) / "results" / "study"
# (key, label, per-cell value from an npz). Lower is better for all but none
# are scored; they are only compared with the baseline.
METRICS = [
    (
        "tracking_error",
        "speed tracking error |measured/commanded - 1|",
        lambda d: np.abs(d["speed_ratio"] - 1.0),
    ),
    ("fall_rate", "fall rate", lambda d: d["fall_rate"]),
    ("peak_force", "peak landing force [x body weight]", lambda d: d["peak_force"]),
    ("unevenness", "uneven load [CV]", lambda d: d["unevenness"]),
    ("turn_rate", "turn rate [deg/s]", lambda d: d["turn_rate"]),
    ("sideways", "sideways drift [m per 10 s]", lambda d: d["sideways"]),
    (
        "speed_fluctuation",
        "speed fluctuation in a stride [RMS / cmd]",
        lambda d: d["speed_fluctuation"],
    ),
    ("bounce", "vertical bounce [RMS m/s]", lambda d: d["bounce"]),
    ("roll_sway", "roll sway [deg, SD]", lambda d: d["roll_sway"]),
    ("pitch_sway", "pitch sway [deg, SD]", lambda d: d["pitch_sway"]),
    ("roll_lean", "mean roll [deg, + = FR side down]", lambda d: d["roll_lean"]),
]
# One line per sweep, keyed by the weight it changes (the tracking sweep
# scales tracking_lin_vel and heading together, named after the former).
SWEEP_KEYS = {
    "gait": "gait",
    "tracking": "tracking_lin_vel",
    "even load": "even_load",
    "peak load": "peak_load",
    # Wide-range study only. roll_sigma is a reward setting, the rest weights.
    "roll tolerance": "roll_sigma",
    "orientation": "orientation",
    "steady speed": "steady_speed",
}
SWEEP_COLORS = {
    "gait": "#2a78d6",
    "tracking": "#eb6834",
    "even load": "#1baf7a",
    "peak load": "#eda100",
    "roll tolerance": "#8a5cc7",
    "orientation": "#c2477f",
    "steady speed": "#5b6b7a",
}
# Baseline values of the swept quantities that STUDY_BASE_WEIGHTS leaves out
# (go2osc_config: reward_settings.roll_sigma, weights.orientation).
BASE_VALUES = {**STUDY_BASE_WEIGHTS, "roll_sigma": 0.25, "orientation": 1.0}
INK, MUTED, GRID = "#1a1a19", "#6b6a64", "#e4e3dd"


def get_args(argv=None):
    parser = argparse.ArgumentParser(description="Summarize the reward study")
    parser.add_argument("--results", type=str, default=str(DATA), help="data")
    parser.add_argument(
        "--out", type=str, default=None, help="figures (default: results/study[/wide])"
    )
    parser.add_argument("--convergence_checkpoint", type=int, default=1250)
    parser.add_argument(
        "--wide",
        action="store_true",
        help="the wide-range study (go2osc_wide_*: to 4.5 m/s and 6 Hz)",
    )
    args = parser.parse_args(argv)
    if args.out is None:
        args.out = str(FIGURES / "wide" if args.wide else FIGURES)
    return args


def use_wide():
    """Point this module (and study_figures, which imports it) at the
    wide-range study: its task prefix and settings."""
    global PREFIX, STUDY_SETTINGS
    PREFIX = "go2osc_wide_"
    STUDY_SETTINGS = WIDE_SETTINGS


def swept(changed, key):
    """Value a setting gives `key`, looking inside reward_settings too."""
    if key in changed:
        return changed[key]
    return changed.get("reward_settings", {}).get(key)


def n_cells(runs):
    d = next(iter(runs.values()))[0][1]
    return len(d["speeds"]) * len(d["omegas"])


def sweep_of(setting):
    """(sweep name, level, weight): level 0 is the baseline, then 1, 2, ...
    in order of the swept weight."""
    changed = STUDY_SETTINGS[setting]
    if not changed:
        return None, 0, None
    for sweep, key in SWEEP_KEYS.items():
        if swept(changed, key) is not None:
            break
    else:
        raise ValueError(f"setting {setting!r} changes no known sweep weight")
    value = swept(changed, key)
    return sweep, sweep_weights(sweep).index(value), value


def sweep_weights(sweep):
    """Weights along a sweep, starting with the baseline's."""
    key = SWEEP_KEYS[sweep]
    levels = sorted(
        swept(w, key) for w in STUDY_SETTINGS.values() if swept(w, key) is not None
    )
    return [BASE_VALUES.get(key, 0.0)] + levels


def load(results, suffix=""):
    """{setting: [(run, npz)]} for files <task>__<run><suffix>.npz."""
    runs = defaultdict(list)
    pattern = re.compile(rf"{PREFIX}(.+?)__(.+?){re.escape(suffix)}\.npz$")
    for path in sorted((Path(results) / "runs").glob(f"{PREFIX}*.npz")):
        match = pattern.fullmatch(path.name)
        if match is None or (not suffix and "__it" in match.group(2)):
            continue
        setting, run = match.groups()
        if setting in STUDY_SETTINGS:
            runs[setting].append((run, dict(np.load(path, allow_pickle=True))))
    return runs


def comparable(early, runs):
    """The earlier-checkpoint results whose gait labels match the final
    results' (results from an older classifier cannot be compared)."""
    out, skipped = {}, 0
    for setting, group in early.items():
        final = dict(runs.get(setting, []))
        keep = [
            (run, d)
            for run, d in group
            if run in final and list(d["gait_names"]) == list(final[run]["gait_names"])
        ]
        skipped += len(group) - len(keep)
        if keep:
            out[setting] = keep
    if skipped:
        print(
            f"Convergence: skipped {skipped} earlier-checkpoint results with "
            "different gait labels (older classifier); re-evaluate them to compare."
        )
    return out


def shares(d):
    """Gait shares [omega, speed, gait], each cell summing to 1."""
    counts = d["gait_counts"]
    return counts / np.maximum(counts.sum(axis=-1, keepdims=True), 1)


def tv_distance(a, b):
    """Mean over cells of the total-variation distance of two share grids."""
    return float(np.mean(0.5 * np.abs(a - b).sum(axis=-1)))


def seed_agreement(runs):
    majority = [shares(d).argmax(axis=-1) for _, d in runs]
    if len(majority) < 2:
        return np.nan
    pairs = [np.mean(a == b) for a, b in itertools.combinations(majority, 2)]
    return float(np.mean(pairs))


def entropy_bits(pooled):
    p = np.clip(pooled, 1e-12, 1)
    return float(np.mean(-(pooled * np.log2(p)).sum(axis=-1)))


def permutation_p(a, b, statistic):
    """Exact two-sided permutation p-value of statistic(a, b) over every
    split of a + b into groups of len(a) and len(b)."""
    pooled = list(a) + list(b)
    observed = statistic(a, b)
    null = [
        statistic(
            [pooled[k] for k in idx],
            [pooled[k] for k in range(len(pooled)) if k not in idx],
        )
        for idx in itertools.combinations(range(len(pooled)), len(a))
    ]
    return float(np.mean(np.abs(null) >= abs(observed) - 1e-12)), null


def run_metric(d, key):
    fn = dict((k, f) for k, _, f in METRICS)[key]
    try:
        return float(np.nanmean(fn(d)))
    except KeyError:  # older results without this metric
        return np.nan


def summarize(runs):
    base = runs.get("baseline", [])
    rows = []
    for setting in STUDY_SETTINGS:
        group = runs.get(setting, [])
        if not group:
            continue
        sweep, level, _ = sweep_of(setting)
        pooled = np.mean([shares(d) for _, d in group], axis=0)
        row = {
            "setting": setting,
            "sweep": sweep or "baseline",
            "level": level,
            "seeds": len(group),
            "seed_agreement": seed_agreement(group),
            "gait_entropy_bits": entropy_bits(pooled),
        }
        if base and setting != "baseline":

            def stat(a, b):
                return tv_distance(np.mean(a, axis=0), np.mean(b, axis=0))

            row["gait_distance"] = stat(
                [shares(d) for _, d in group], [shares(d) for _, d in base]
            )
            p, null = permutation_p(
                [shares(d) for _, d in group], [shares(d) for _, d in base], stat
            )
            row["gait_distance_p"] = p
            row["gait_distance_null95"] = float(np.percentile(null, 95))
        for key, _, _ in METRICS:
            values = np.array([run_metric(d, key) for _, d in group])
            row[f"{key}_mean"] = float(np.nanmean(values))
            row[f"{key}_sd"] = (
                float(np.nanstd(values, ddof=1)) if len(values) > 1 else np.nan
            )
            if base and setting != "baseline":
                ref = np.array([run_metric(d, key) for _, d in base])
                if not (np.isnan(values).any() or np.isnan(ref).any()):
                    row[f"{key}_p"], _ = permutation_p(
                        values, ref, lambda a, b: np.mean(a) - np.mean(b)
                    )
        rows.append(row)
    return rows


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def plot_metrics(rows, runs, path):
    """Each metric against the weight level, one line per sweep; every
    sweep starts from the shared baseline (level 0)."""
    panels = [
        ("gait_distance", "gait change from baseline [TV distance]"),
        ("seed_agreement", "seed agreement (same majority gait)"),
    ]
    panels += [(key, label) for key, label, _ in METRICS]
    by = {r["setting"]: r for r in rows}
    base = by.get("baseline")
    ncols = 4
    nrows = -(-len(panels) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(17, 3.75 * nrows))
    for ax in axes.ravel()[len(panels) :]:
        ax.axis("off")
    for ax, (key, label) in zip(axes.ravel(), panels):
        style(ax)
        for sweep, color in SWEEP_COLORS.items():
            pts = [r for r in rows if r["sweep"] == sweep]
            if not pts:  # sweep not trained (yet): don't draw its baseline point
                continue
            if base is not None:
                pts = [base] + pts
            pts = sorted(pts, key=lambda r: r["level"])
            if not pts:
                continue
            x = [r["level"] for r in pts]
            if key == "gait_distance":
                y = [r.get("gait_distance", 0.0) for r in pts]
                err = None
            elif key == "seed_agreement":
                y, err = [r["seed_agreement"] for r in pts], None
            else:
                y = [r[f"{key}_mean"] for r in pts]
                err = [r[f"{key}_sd"] for r in pts]
            ax.errorbar(
                x,
                y,
                yerr=err,
                color=color,
                lw=2,
                marker="o",
                ms=7,
                capsize=3,
                label=f"{sweep}: " + " → ".join(f"{w:g}" for w in sweep_weights(sweep)),
                markeredgecolor="white",
                markeredgewidth=1.5,
            )
            # Mark settings that differ from the baseline (permutation p < 0.05).
            pkey = "gait_distance_p" if key == "gait_distance" else f"{key}_p"
            for r, xi, yi in zip(pts, x, y):
                if r.get(pkey, 1.0) < 0.05:
                    ax.annotate(
                        "*",
                        (xi, yi),
                        textcoords="offset points",
                        xytext=(6, 4),
                        color=INK,
                        fontsize=12,
                    )
        if key == "gait_distance":
            null = [
                r["gait_distance_null95"] for r in rows if "gait_distance_null95" in r
            ]
            if null:
                ax.axhspan(
                    0,
                    np.mean(null),
                    color=GRID,
                    alpha=0.7,
                    lw=0,
                    label="seed-to-seed range (95%)",
                )
        if base is not None and key not in ("gait_distance", "seed_agreement"):
            sd = base[f"{key}_sd"]
            if not np.isnan(sd):
                m = base[f"{key}_mean"]
                ax.axhspan(
                    m - sd, m + sd, color=GRID, alpha=0.7, lw=0, label="baseline ± SD"
                )
        ax.set_title(label, fontsize=9, color=INK)
        ax.set_xticks([0, 1, 2], ["baseline", "level 1", "level 2"])
        ax.set_xlabel("reward weight level (legend: weights)", fontsize=8, color=MUTED)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    extra = axes.ravel()[2].get_legend_handles_labels()
    for handle, label in zip(*extra):
        if label not in labels:
            handles.append(handle)
            labels.append(label)
    fig.legend(handles, labels, loc="upper right", frameon=False, fontsize=9)
    fig.suptitle(
        "Reward study, front-right locked: each sweep raises one reward weight from "
        "the shared baseline.\nPoints: mean over seeds; bars: SD; * : differs from "
        "baseline (exact permutation test, p < 0.05)",
        fontsize=10,
        color=INK,
        x=0.02,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=120)
    print(f"Saved {path}")


def plot_gait_maps(runs, path):
    """Majority gait (pooled over seeds) per cell for every setting, with the
    pooled share and the seed agreement."""
    settings = [s for s in STUDY_SETTINGS if runs.get(s)]
    any_d = runs[settings[0]][0][1]
    names = list(any_d["gait_names"])
    speeds, omegas = any_d["speeds"], any_d["omegas"]
    colors = gait_colors(names)
    ncols = min(5, len(settings))
    nrows = -(-len(settings) // ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.2 * ncols + 1.6, 4.0 * nrows + 0.4), squeeze=False
    )
    spare = axes.ravel()[len(settings) :]
    for ax in spare:
        ax.axis("off")
    for ax, setting in zip(axes.ravel(), settings):
        group = runs[setting]
        pooled = np.mean([shares(d) for _, d in group], axis=0)
        majority = [shares(d).argmax(axis=-1) for _, d in group]
        dominant = pooled.argmax(axis=-1)
        ax.imshow(
            dominant,
            cmap=ListedColormap(colors),
            vmin=-0.5,
            vmax=len(names) - 0.5,
            origin="lower",
            aspect="auto",
        )
        for (i, j), g in np.ndenumerate(dominant):
            agree = (
                np.mean([m[i, j] == dominant[i, j] for m in majority])
                if majority
                else np.nan
            )
            rgb = matplotlib.colors.to_rgb(colors[g])
            light = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2] > 0.5
            ax.text(
                j,
                i,
                f"{pooled[i, j, g]:.0%}\n{agree * len(group):.0f}/{len(group)}",
                ha="center",
                va="center",
                fontsize=7,
                color=INK if light else "white",
            )
        ax.set_title(f"{setting} ({len(group)} seeds)", fontsize=10, color=INK)
        ax.set_xticks(range(len(speeds)), [f"{v:g}" for v in speeds], fontsize=8)
        ax.set_yticks(range(len(omegas)), [f"{w:g}" for w in omegas], fontsize=8)
        ax.set_xlabel("commanded speed [m/s]", fontsize=8)
        ax.set_ylabel("oscillator frequency [Hz]", fontsize=8)
    handles, seen = observed_legend(
        names, [d["gait_counts"] for s in settings for _, d in runs[s]]
    )
    fig.suptitle(
        "Majority gait pooled over seeds. Cell text: pooled share of robots; "
        "seeds whose own majority gait matches",
        fontsize=10,
        color=INK,
    )
    if len(spare):  # legend in the first empty panel
        spare[0].legend(handles, seen, loc="center left", frameon=False, fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
    else:
        fig.legend(handles, seen, loc="center right", frameon=False, fontsize=8)
        fig.tight_layout(rect=(0, 0, 0.88, 0.93))
    fig.savefig(path, dpi=120)
    print(f"Saved {path}")


# Shades within a gait family (gait-mix maps): the hop by its mean rear-leg
# lead (dark: rear legs land after the front leg; light: before), the two-leg
# gaits by how much the third leg is used (dark: held up; light: every stride).
# One scale for the hop and the bound, by rear-leg lead (how far the rear
# pair lands before the front leg, in strides; circular, so -0.5 and +0.5
# are the same): bound with the front leg well ahead in dark purple, the hop
# range from dark to light blue, bound with the rear legs well ahead in light
# lavender. Lighter = rear legs earlier.
_LEAD_NODES = [
    (-0.5, "#1e0a3c"),
    (-HOP_MAX_REAR_LAG, "#0b3d91"),
    (HOP_MAX_REAR_LEAD, "#b3d6f7"),
    (0.5, "#e6dcf5"),
]
LEAD_SHADE = (
    matplotlib.colors.LinearSegmentedColormap.from_list(
        "lead", [(v + 0.5, c) for v, c in _LEAD_NODES]
    ),
    matplotlib.colors.Normalize(-0.5, 0.5),
    "hop / bound: rear lead\nhop = inside",
)
USE_SHADE = (
    matplotlib.colors.LinearSegmentedColormap.from_list("use", ["#7f0f22", "#f9c9d1"]),
    matplotlib.colors.Normalize(0.0, 1.0),
    "2-leg: 3rd leg use",
)
USE_LABELS = ["2-leg, together", "2-leg + sporadic 3rd", "2-leg hop + 3rd-leg support"]


def shades(ds, names):
    """{label: (per-cell mean, shade)} from the per-robot values of the npz
    dicts in `ds` (pooled): the hop's rear-leg lead and the two-leg gaits'
    third-leg use. Empty for results without per-robot labels."""
    if not all("robot_label" in d for d in ds):
        return {}
    label = np.concatenate([d["robot_label"] for d in ds], axis=-1)
    out = {}
    for name, key, shade in [
        ("hop", "robot_lead", LEAD_SHADE),
        ("bound", "robot_lead", LEAD_SHADE),
    ] + [(n, "robot_third_use", USE_SHADE) for n in USE_LABELS]:
        if name not in names:
            continue
        value = np.concatenate([d[key] for d in ds], axis=-1)
        mask = label == names.index(name)
        with np.errstate(invalid="ignore"):
            mean = np.where(mask, value, 0).sum(-1) / mask.sum(-1)
        out[name] = (mean, shade)
    return out


def gait_mix(ax, share, names, label_min=0.2, shade=None):
    """Each cell split left to right into one segment per gait, its width the
    gait's share of robots (share: [omega, speed, gait], rows summing to 1).
    Gaits keep the same order in every cell so a gait can be followed across
    cells; segments of at least `label_min` carry their share. With `shade`
    (from shades()), the hop and two-leg segments are coloured by their
    per-cell value, which is also printed. Returns the shades used."""
    colors = gait_colors(names)
    shade = shade or {}
    used = {}
    rows, cols = share.shape[:2]
    for i in range(rows):
        for j in range(cols):
            x = j - 0.5
            if not share[i, j].sum():
                ax.add_patch(
                    plt.Rectangle(
                        (x, i - 0.5),
                        1,
                        1,
                        facecolor="white",
                        edgecolor=GRID,
                        hatch="//",
                    )
                )
                continue
            for g, s in enumerate(share[i, j]):
                if s <= 0:
                    continue
                face, note = colors[g], ""
                if names[g] in shade and not np.isnan(shade[names[g]][0][i, j]):
                    value, (cmap, norm, title) = (
                        shade[names[g]][0][i, j],
                        shade[names[g]][1],
                    )
                    face = cmap(norm(value))
                    used[title] = (cmap, norm)
                    note = (
                        f"\n{value:+.2f}"
                        if names[g] in ("hop", "bound")
                        else f"\n{value:.0%} use"
                    )
                ax.add_patch(
                    plt.Rectangle(
                        (x, i - 0.5),
                        s,
                        1,
                        facecolor=face,
                        edgecolor="white",
                        linewidth=0.4,
                    )
                )
                if s >= label_min:
                    rgb = matplotlib.colors.to_rgb(face)
                    light = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2] > 0.5
                    ax.text(
                        x + s / 2,
                        i,
                        f"{s:.0%}{note}",
                        ha="center",
                        va="center",
                        fontsize=6.5,
                        color=INK if light else "white",
                    )
                x += s
    for j in range(cols + 1):
        ax.axvline(j - 0.5, color="white", lw=3)
    for i in range(rows + 1):
        ax.axhline(i - 0.5, color="white", lw=3)
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(-0.5, rows - 0.5)
    return used


def mark_hop_range(bar):
    """Lines at the hop's limits on a rear-lead colour bar (bound outside)."""
    for v in (-HOP_MAX_REAR_LAG, HOP_MAX_REAR_LEAD):
        bar.ax.axhline(v, color=INK, lw=0.8)
    bar.set_ticks([-0.5, -HOP_MAX_REAR_LAG, 0, HOP_MAX_REAR_LEAD, 0.5])
    bar.ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%+.2f"))


def shade_bars(fig, used, left, bottom, height=0.3, step=0.06):
    """Small vertical colour bars for the shades a gait-mix map used, side by
    side from figure position (left, bottom)."""
    for k, (title, (cmap, norm)) in enumerate(sorted(used.items())):
        cax = fig.add_axes([left + k * step, bottom, 0.012, height])
        bar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
        bar.ax.tick_params(labelsize=7)
        if "use" in title:
            bar.ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        else:
            mark_hop_range(bar)
        cax.set_title(title.replace(": ", ":\n"), fontsize=7, loc="left")


def plot_gait_mix_maps(runs, path):
    """Gait mix per cell, pooled over seeds, for every setting."""
    settings = [s for s in STUDY_SETTINGS if runs.get(s)]
    any_d = runs[settings[0]][0][1]
    names = list(any_d["gait_names"])
    speeds, omegas = any_d["speeds"], any_d["omegas"]
    ncols = min(3, len(settings))
    nrows = -(-len(settings) // ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6.2 * ncols + 1.8, 4.6 * nrows), squeeze=False
    )
    for ax in axes.ravel()[len(settings) :]:
        ax.axis("off")
    used = {}
    for ax, setting in zip(axes.ravel(), settings):
        pooled = np.mean([shares(d) for _, d in runs[setting]], axis=0)
        used.update(
            gait_mix(
                ax,
                pooled,
                names,
                label_min=0.25,
                shade=shades([d for _, d in runs[setting]], names),
            )
        )
        ax.set_title(f"{setting} ({len(runs[setting])} seeds)", fontsize=10, color=INK)
        ax.set_xticks(range(len(speeds)), [f"{v:g}" for v in speeds], fontsize=8)
        ax.set_yticks(range(len(omegas)), [f"{w:g}" for w in omegas], fontsize=8)
        ax.set_xlabel("commanded speed [m/s]", fontsize=8)
        ax.set_ylabel("oscillator frequency [Hz]", fontsize=8)
    handles, seen = observed_legend(
        names, [d["gait_counts"] for s in settings for _, d in runs[s]]
    )
    fig.suptitle(
        "Gait mix per cell, pooled over seeds: segment width = share of robots "
        "(same gait order in every cell; shares of 25% or more labelled)",
        fontsize=10,
        color=INK,
    )
    spare = axes.ravel()[len(settings) :]
    if len(spare):
        # Legend and colour bars go in the first empty panel.
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        box = spare[0].get_position()
        spare[0].legend(handles, seen, loc="upper left", frameon=False, fontsize=9)
        shade_bars(
            fig,
            used,
            left=box.x0 + 0.55 * box.width,
            bottom=box.y0 + 0.05 * box.height,
            height=0.6 * box.height,
            step=0.2 * box.width,
        )
    else:
        fig.legend(handles, seen, loc="center right", frameon=False, fontsize=9)
        fig.tight_layout(rect=(0, 0, 0.86, 0.95))
        shade_bars(fig, used, left=0.875, bottom=0.08)
    fig.savefig(path, dpi=120)
    print(f"Saved {path}")


def plot_gait_shares(runs, path):
    """Share of robots in each gait over the whole grid, per setting and seed."""
    settings = [s for s in STUDY_SETTINGS if runs.get(s)]
    names = list(runs[settings[0]][0][1]["gait_names"])
    colors = gait_colors(names)
    labels, rows = [], []
    for setting in settings:
        for run, d in runs[setting]:
            labels.append(f"{setting}  {run.split('_')[-1]}")
            rows.append(shares(d).reshape(-1, len(names)).mean(axis=0))
        labels.append(f"{setting}  pooled")
        rows.append(np.mean(rows[-len(runs[setting]) :], axis=0))
    rows = np.array(rows)
    fig, ax = plt.subplots(figsize=(9, 0.28 * len(labels) + 1.5))
    left = np.zeros(len(labels))
    y = np.arange(len(labels))[::-1]
    for g, name in enumerate(names):
        ax.barh(
            y,
            rows[:, g],
            left=left,
            color=colors[g],
            height=0.8,
            edgecolor="white",
            linewidth=1,
            label=name if rows[:, g].any() else None,
        )
        left += rows[:, g]
    ax.set_yticks(y, labels, fontsize=7)
    for tick, label in zip(ax.get_yticklabels(), labels):
        if label.endswith("pooled"):
            tick.set_fontweight("bold")
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=8)
    ax.set_title(
        f"Share of robots in each gait over all {n_cells(runs)} cells",
        fontsize=10,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"Saved {path}")


def main():
    args = get_args()
    if args.wide:
        use_wide()
    # Metrics missing from older results are NaN; their empty means are expected.
    warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
    warnings.filterwarnings("ignore", "Degrees of freedom", RuntimeWarning)
    results = Path(args.results)
    runs = load(results)
    if not runs:
        raise SystemExit(f"no {PREFIX}*.npz under {results}; run study_eval.py first")
    rows = summarize(runs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    columns = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "study_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {out / 'study_summary.csv'}\n")

    print(
        f"{'setting':10s} seeds  gait-change (p)   agree  entropy  "
        + "  ".join(f"{k:>14s}" for k, _, _ in METRICS)
    )
    for r in rows:
        change = (
            f"{r['gait_distance']:.2f} ({r['gait_distance_p']:.3f})"
            if "gait_distance" in r
            else "      -      "
        )
        cells = "  ".join(
            f"{r[f'{k}_mean']:6.2f}±{r[f'{k}_sd']:<5.2f}"
            + ("*" if r.get(f"{k}_p", 1) < 0.05 else " ")
            for k, _, _ in METRICS
        )
        print(
            f"{r['setting']:10s} {r['seeds']:5d}  {change:15s}  "
            f"{r['seed_agreement']:5.2f}  {r['gait_entropy_bits']:7.2f}  {cells}"
        )

    # Convergence: did each run's gait still change after the earlier checkpoint?
    early = comparable(load(results, suffix=f"__it{args.convergence_checkpoint}"), runs)
    if early:
        print(f"\nGait change from iteration {args.convergence_checkpoint} to final:")
        for setting, group in early.items():
            final = dict(runs.get(setting, []))
            for run, d in group:
                if run in final:
                    print(
                        f"  {setting:10s} {run}: "
                        f"{tv_distance(shares(d), shares(final[run])):.2f}"
                    )

    plot_metrics(rows, runs, out / "study_metrics.png")
    plot_gait_maps(runs, out / "study_gait_maps.png")
    plot_gait_mix_maps(runs, out / "study_gait_mix_maps.png")
    plot_gait_shares(runs, out / "study_gait_shares.png")


if __name__ == "__main__":
    main()
