"""Summarize the reward study: how each setting changes the gaits and metrics.

Reads logs/study_results/go2osc_study_<setting>__<run>.npz (from
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

Writes study_summary.csv, study_metrics.png, study_gait_maps.png and
study_gait_shares.png next to the results.

Usage:
    uv run --frozen scripts/study_summary.py
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

from grid_eval import gait_colors
from gym.envs.go2.go2osc_config import STUDY_BASE_WEIGHTS, STUDY_SETTINGS
from gym.utils.helpers import GYM_ROOT_DIR

PREFIX = "go2osc_study_"
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
]
# One line per sweep, keyed by the weight it changes (the tracking sweep
# scales tracking_lin_vel and heading together, named after the former).
SWEEP_KEYS = {
    "gait": "gait",
    "tracking": "tracking_lin_vel",
    "even load": "even_load",
    "peak load": "peak_load",
}
SWEEP_COLORS = {
    "gait": "#2a78d6",
    "tracking": "#eb6834",
    "even load": "#1baf7a",
    "peak load": "#eda100",
}
INK, MUTED, GRID = "#1a1a19", "#6b6a64", "#e4e3dd"


def get_args(argv=None):
    parser = argparse.ArgumentParser(description="Summarize the reward study")
    parser.add_argument(
        "--results",
        type=str,
        default=str(Path(GYM_ROOT_DIR) / "logs" / "study_results"),
    )
    parser.add_argument("--convergence_checkpoint", type=int, default=1250)
    return parser.parse_args(argv)


def sweep_of(setting):
    """(sweep name, level, weight): level 0 is the baseline, then 1, 2, ...
    in order of the swept weight."""
    changed = STUDY_SETTINGS[setting]
    if not changed:
        return None, 0, None
    for sweep, key in SWEEP_KEYS.items():
        if key in changed:
            break
    else:
        raise ValueError(f"setting {setting!r} changes no known sweep weight")
    return sweep, sweep_weights(sweep).index(changed[key]), changed[key]


def sweep_weights(sweep):
    """Weights along a sweep, starting with the baseline's."""
    key = SWEEP_KEYS[sweep]
    levels = sorted(w[key] for w in STUDY_SETTINGS.values() if key in w)
    return [STUDY_BASE_WEIGHTS.get(key, 0.0)] + levels


def load(results, suffix=""):
    """{setting: [(run, npz)]} for files <task>__<run><suffix>.npz."""
    runs = defaultdict(list)
    pattern = re.compile(rf"{PREFIX}(.+?)__(.+?){re.escape(suffix)}\.npz$")
    for path in sorted(Path(results).glob(f"{PREFIX}*.npz")):
        match = pattern.fullmatch(path.name)
        if match is None or (not suffix and "__it" in match.group(2)):
            continue
        setting, run = match.groups()
        if setting in STUDY_SETTINGS:
            runs[setting].append((run, dict(np.load(path, allow_pickle=True))))
    return runs


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
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.5))
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
    fig, axes = plt.subplots(
        1, len(settings), figsize=(4.2 * len(settings) + 1.6, 4.4), squeeze=False
    )
    for ax, setting in zip(axes[0], settings):
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
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]
    fig.legend(handles, names, loc="center right", frameon=False, fontsize=8)
    fig.suptitle(
        "Majority gait pooled over seeds. Cell text: pooled share of robots; "
        "seeds whose own majority gait matches",
        fontsize=10,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 0.88, 0.93))
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
            label=name,
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
        "Share of robots in each gait over all 20 cells", fontsize=10, color=INK
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"Saved {path}")


def main():
    args = get_args()
    # Metrics missing from older results are NaN; their empty means are expected.
    warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
    warnings.filterwarnings("ignore", "Degrees of freedom", RuntimeWarning)
    results = Path(args.results)
    runs = load(results)
    if not runs:
        raise SystemExit(f"no {PREFIX}*.npz under {results}; run study_eval.py first")
    rows = summarize(runs)

    columns = list(dict.fromkeys(k for r in rows for k in r))
    with open(results / "study_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {results / 'study_summary.csv'}\n")

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
    early = load(results, suffix=f"__it{args.convergence_checkpoint}")
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

    plot_metrics(rows, runs, results / "study_metrics.png")
    plot_gait_maps(runs, results / "study_gait_maps.png")
    plot_gait_shares(runs, results / "study_gait_shares.png")


if __name__ == "__main__":
    main()
