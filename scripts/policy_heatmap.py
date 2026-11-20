"""Slide heat map of one policy over the wide grid (speed x oscillator
frequency): speed tracking, fall rate, dominant gait and joint power, drawn
from the saved study_eval.py results (no new simulation).

Usage:
    uv run --frozen scripts/policy_heatmap.py go2osc_wide_even2.5_deploy_track Nov16_16-13-14_seed2 \
        [--title "deploy_track seed 2"] [--train_max 2.5] [--out results/study/wide/slides/x.png]
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "logs" / "study_results" / "runs"
GAIT_GROUPS = [  # (label, color, gait names), as slide_gait_shares.py
    ("Pronk", "#2a6fd6", ["hop", "hop, staggered landing"]),
    ("Bound", "#8fb8ef", ["bound"]),
    ("Two-leg", "#d4521f", ["2-leg, together", "2-leg, alternating"]),
    ("Two-leg + light 3rd leg", "#f0ad85", None),  # other, least-loaded leg < 13%
    ("Three-leg", "#1ba57a", None),  # other periodic labels
    ("Irregular", "#c6c5ba", ["irregular", "not stepping"]),
]
LIGHT = 0.13  # least-loaded leg's load share below which a third leg is "light"
MOSTLY_FELL = 0.95  # cells where at least this share fell are left blank
INK, MUTED = "#1a1a19", "#77776f"


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("task")
    p.add_argument("run")
    p.add_argument("--title", default=None)
    p.add_argument(
        "--train_max",
        type=float,
        default=None,
        help="max trained forward command [m/s]; faster columns are hatched",
    )
    p.add_argument("--out", default=None)
    a = p.parse_args()
    from matplotlib import font_manager

    for f in glob.glob("/usr/share/fonts/truetype/lato/Lato-*.ttf"):
        font_manager.fontManager.addfont(f)
    plt.rcParams["font.family"] = ["Lato", "DejaVu Sans"]

    z = np.load(RUNS / f"{a.task}__{a.run}.npz", allow_pickle=False)
    sp, om = z["speeds"], z["omegas"]
    names = list(z["gait_names"])
    # Per-robot groups (fallen robots have label -1), then each cell's
    # most common group and its share.
    label = z["robot_label"]
    group = np.full(label.shape, -1)
    for k, (_, _, members) in enumerate(GAIT_GROUPS):
        if members:
            group[np.isin(label, [names.index(m) for m in members])] = k
    other = (label >= 0) & (group < 0)
    light = np.nanmin(z["robot_leg_shares"], -1) < LIGHT
    group[other & light] = 3
    group[other & ~light] = 4
    group_counts = np.stack([(group == k).sum(-1) for k in range(len(GAIT_GROUPS))], -1)
    total = group_counts.sum(-1)
    gone = z["fall_rate"] >= MOSTLY_FELL
    dom = np.where((total > 0) & ~gone, group_counts.argmax(-1), -1)
    dom_share = np.where(total > 0, group_counts.max(-1) / np.maximum(total, 1), np.nan)

    panels = [
        ("Measured / commanded speed", np.where(gone, np.nan, z["speed_ratio"]), "RdBu", (0.5, 1.5), "{:.2f}"),
        ("Fall rate", z["fall_rate"], "Reds", (0, 1), "{:.0%}"),
        ("Dominant gait", None, None, None, None),
        ("Joint power [W]", np.where(gone, np.nan, z["power"]), "viridis", None, "{:.0f}"),
    ]
    fig, axs = plt.subplots(1, 4, figsize=(20, 5.6), dpi=170)
    ext = (-0.5, len(sp) - 0.5, -0.5, len(om) - 0.5)
    for ax, (title, data, cmap, lim, fmt) in zip(axs, panels):
        if data is None:
            cm = ListedColormap([g[1] for g in GAIT_GROUPS])
            shown = np.ma.masked_less(dom, 0)
            ax.imshow(
                shown,
                origin="lower",
                cmap=cm,
                vmin=-0.5,
                vmax=len(GAIT_GROUPS) - 0.5,
                extent=ext,
                aspect="auto",
            )
            for i in range(len(om)):
                for j in range(len(sp)):
                    if dom[i, j] >= 0:
                        dark = GAIT_GROUPS[dom[i, j]][0] in (
                            "Pronk",
                            "Two-leg",
                            "Three-leg",
                        )
                        ax.text(
                            j,
                            i,
                            f"{dom_share[i, j]:.0%}",
                            ha="center",
                            va="center",
                            fontsize=8,
                            color="white" if dark else INK,
                        )
            present = sorted(set(dom[dom >= 0].ravel()))
            ax.legend(
                [Patch(color=GAIT_GROUPS[k][1]) for k in present],
                [GAIT_GROUPS[k][0] for k in present],
                loc="upper center",
                bbox_to_anchor=(0.5, -0.13),
                ncol=len(present),
                frameon=False,
                fontsize=9,
            )
        else:
            vmin, vmax = lim if lim else (np.nanmin(data), np.nanmax(data))
            im = ax.imshow(
                np.ma.masked_invalid(data),
                origin="lower",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                extent=ext,
                aspect="auto",
            )
            for i in range(len(om)):
                for j in range(len(sp)):
                    v = data[i, j]
                    if np.isfinite(v):
                        frac = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                        strong = (
                            abs(frac - 0.5) > 0.3
                            if cmap == "RdBu"
                            else (frac > 0.55 if cmap == "Reds" else frac < 0.45)
                        )
                        ax.text(
                            j,
                            i,
                            fmt.format(v),
                            ha="center",
                            va="center",
                            fontsize=8,
                            color="white" if strong else INK,
                        )
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        ax.set_xticks(range(len(sp)), [f"{s:g}" for s in sp])
        ax.set_yticks(range(len(om)), [f"{o:g}" for o in om])
        ax.set_xlabel("Commanded speed [m/s]", color=INK)
        ax.set_ylabel("Oscillator frequency [Hz]", color=INK)
        ax.set_title(title, fontsize=13, fontweight="bold", color=INK)
        if a.train_max is not None:
            j0 = int(np.searchsorted(sp, a.train_max, side="right"))
            if j0 < len(sp):
                ax.add_patch(
                    Rectangle(
                        (j0 - 0.5, -0.5),
                        len(sp) - j0,
                        len(om),
                        fill=False,
                        hatch="///",
                        edgecolor="#55554f",
                        linewidth=0,
                        alpha=0.5,
                    )
                )
                ax.axvline(j0 - 0.5, color=INK, linewidth=1.5, linestyle="--")
    head = a.title or f"{a.task} {a.run}"
    sub = "64 robots per cell, measured 10 s after 10 s settling; blank: ≥ 95% fell"
    if a.train_max is not None:
        sub += (
            f"; hatched: faster than the trained command range (≤ {a.train_max:g} m/s)"
        )
    fig.suptitle(head, fontsize=16, fontweight="bold", color=INK, y=0.99)
    fig.text(0.5, 0.925, sub, ha="center", color=MUTED, fontsize=10.5)
    fig.subplots_adjust(left=0.04, right=0.98, top=0.85, bottom=0.17, wspace=0.32)
    out = Path(
        a.out or ROOT / "results/study/wide/slides" / f"heatmap_{a.task}__{a.run}.png"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
